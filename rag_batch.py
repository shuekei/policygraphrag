import os
import json
import time
from datetime import datetime
from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, AutoModel
import requests
import psutil
import threading
import csv

# ============================================================
# NEO4J CONNECTION
# ============================================================

NEO4J_URI="neo4j+s://74693ba9.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="PZFuTUEZblB5xi9JbxgEob1x0ZJ0qBdbWJiuKlaXQLw"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("✅ Neo4j connected.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to(device)

def get_embedding(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)

    if hasattr(out, "embeddings") and out.embeddings is not None:
        return out.embeddings.squeeze(0).cpu().numpy().tolist()

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output.squeeze(0).cpu().numpy().tolist()

    hidden = out.last_hidden_state
    emb = hidden.mean(dim=1)
    return emb.squeeze(0).cpu().numpy().tolist()

# system monitoring
class SystemMonitor:
    def __init__(self, csv_path, interval=1.0):
        self.csv_path = csv_path
        self.interval = interval
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _run(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","cpu_percent","ram_percent","ram_used_gb",
                "gpu_util_percent","gpu_mem_mib"
            ])

            while self.running:
                ts = datetime.now().strftime("%H:%M:%S")
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                ram_pct = mem.percent
                ram_gb = mem.used / (1024**3)

                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024**2)
                    gpu_util = (torch.cuda.memory_allocated() /
                                torch.cuda.get_device_properties(0).total_memory) * 100
                else:
                    gpu_mem, gpu_util = 0, 0

                writer.writerow([ts, round(cpu,2), round(ram_pct,2),
                                 round(ram_gb,2), round(gpu_util,2),
                                 round(gpu_mem,0)])
                f.flush()
                time.sleep(self.interval)

# ============================================================
# VECTOR SEARCH
# ============================================================

INDEXES = {
    "Clause": "clause_embedding_index",
    "Section": "section_embedding_index",
    "Subsection": "subsection_embedding_index",
    "Part": "part_embedding_index"
}

def vector_search(session, index_name, embedding, k=3):
    query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
    YIELD node, score
    RETURN node, score
    """
    return [{"node": r["node"], "score": r["score"]} for r in session.run(query, k=k, embedding=embedding)]

def retrieve_top_k(embedding, k=3):
    with driver.session() as session:
        hits = []
        for label, idx in INDEXES.items():
            for h in vector_search(session, idx, embedding, k):
                h["label"] = label
                hits.append(h)
        return sorted(hits, key=lambda x: x["score"], reverse=True)[:k]



# ============================================================
# GRAPH EXPANSION
# ============================================================

def expand_node(session, node_id, label):
    if label == "Clause":
        query = """MATCH (c:Clause) WHERE elementId(c)=$id
        OPTIONAL MATCH (s:Section)-[:HAS_CLAUSE]->(c)
        OPTIONAL MATCH (p:Part)-[:HAS_SECTION]->(s)
        OPTIONAL MATCH (sf:SourceFile)-[:HAS_PART]->(p)
        RETURN c, s, p, sf"""
    elif label == "Subsection":
        query = """MATCH (ss:Subsection) WHERE elementId(ss)=$id
        OPTIONAL MATCH (s:Section)-[:HAS_SUBSECTION]->(ss)
        OPTIONAL MATCH (p:Part)-[:HAS_SECTION]->(s)
        OPTIONAL MATCH (sf:SourceFile)-[:HAS_PART]->(p)
        OPTIONAL MATCH (ss)-[:HAS_CLAUSE]->(c:Clause)
        RETURN ss, s, p, sf, collect(c) AS clauses"""
    elif label == "Section":
        query = """MATCH (s:Section) WHERE elementId(s)=$id
        OPTIONAL MATCH (p:Part)-[:HAS_SECTION]->(s)
        OPTIONAL MATCH (sf:SourceFile)-[:HAS_PART]->(p)
        OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss:Subsection)
        OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(c:Clause)
        RETURN s, p, sf, collect(ss) AS subs, collect(c) AS clauses"""
    else:
        query = """MATCH (p:Part) WHERE elementId(p)=$id
        OPTIONAL MATCH (sf:SourceFile)-[:HAS_PART]->(p)
        OPTIONAL MATCH (p)-[:HAS_SECTION]->(s:Section)
        RETURN p, sf, collect(s) AS sections"""

    r = session.run(query, id=node_id).single()
    return r.data() if r else None


# ============================================================
# CONTEXT FORMATTER
# ============================================================

def format_context(label, expanded):
    meta, ctx = [], []

    node = expanded["c"] if label=="Clause" else \
           expanded["ss"] if label=="Subsection" else \
           expanded["s"] if label=="Section" else expanded["p"]

    meta.append("=== SOURCE METADATA ===")
    meta.append(f"File: {node.get('source_file')}")
    meta.append(f"Page: {node.get('page_no')}")
    if node.get("part_title"): meta.append(f"Part: {node['part_title']}")
    if node.get("section_title"): meta.append(f"Section: {node['section_title']}")
    if node.get("subsection_title"): meta.append(f"Subsection: {node['subsection_title']}")
    if node.get("clause_number"): meta.append(f"Clause: {node['clause_number']}")
    meta.append("=======================\n")

    if label == "Clause":
        ctx.append(f"[CLAUSE] {node.get('text')}")

    elif label == "Subsection":
        ctx.append(f"[SUBSECTION] {node.get('title')}")
        for c in expanded["clauses"]:
            ctx.append(f"[CLAUSE] {c.get('text')}")

    elif label == "Section":
        ctx.append(f"[SECTION] {node.get('title')}")
        for ss in expanded["subs"]:
            ctx.append(f"[SUBSECTION] {ss.get('title')}")
        for c in expanded["clauses"]:
            ctx.append(f"[CLAUSE] {c.get('text')}")

    else:
        ctx.append(f"[PART] {node.get('title')}")
        for s in expanded["sections"]:
            ctx.append(f"[SECTION] {s.get('title')}")

    return "\n".join(meta) + "\n" + "\n".join(ctx)


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama_metrics(context, question):

    prompt = f"""
You are a regulatory policy expert. Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}
"""

    payload = {
        "model": "llama3-chatqa:70b",
        "prompt": prompt,
        "stream": True
    }

    timestamp_sent = time.time()
    first_token_time = None
    answer_chunks = []

    r = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    for line in r.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))

            if first_token_time is None:
                first_token_time = time.time()

            if "response" in data:
                answer_chunks.append(data["response"])

            if data.get("done"):
                break

    timestamp_completed = time.time()
    answer = "".join(answer_chunks)

    ttfb = first_token_time - timestamp_sent
    total_latency = timestamp_completed - timestamp_sent
    gen_time = timestamp_completed - first_token_time

    # token estimate
    est_tokens = len(tokenizer.encode(answer))
    tps = est_tokens / gen_time if gen_time > 0 else 0

    return {
        "answer": answer,
        "timestamp_sent": timestamp_sent,
        "timestamp_first_byte": first_token_time,
        "timestamp_completed": timestamp_completed,
        "ttfb": ttfb,
        "gen_time": gen_time,
        "total_latency": total_latency,
        "tokens": est_tokens,
        "tps": tps
    }


# ============================================================
# QUESTIONS (BNM STYLE)
# ============================================================

QUESTIONS = [
    "What is the purpose of this policy document?",
    "What is the applicability of this policy?",
    "Who does this policy apply to?",
    "What are the key objectives of this policy?",
    "When does this policy come into effect?",
    "What are the main regulatory requirements?",
    "What institutions are covered under this policy?",
    "What are the responsibilities of the board of directors?",
    "What are the responsibilities of senior management?",
    "What governance requirements are specified?",
    "What risk management practices are required?",
    "What reporting obligations are stated?",
    "What documentation must be maintained?",
    "What internal controls are required?",
    "What compliance requirements are mentioned?",
    "What are the penalties for non-compliance?",
    "What disclosure requirements are specified?",
    "What approval processes are required?",
    "What operational requirements are outlined?",
    "What record-keeping obligations are stated?",
    "What are the audit requirements?",
    "What outsourcing requirements are included?",
    "What technology risk requirements are mentioned?",
    "What data protection requirements are specified?",
    "What business continuity requirements are stated?",
    "What consumer protection principles are included?",
    "What product governance requirements are described?",
    "What are the requirements for customer due diligence?",
    "What ongoing monitoring obligations are stated?",
    "What escalation procedures are required?",
    "What internal policies must be established?",
    "What training requirements are mentioned?",
    "What review and update obligations are stated?",
    "What risk assessment processes are required?",
    "What oversight mechanisms are described?",
    "What control functions are required?",
    "What responsibilities are assigned to compliance functions?",
    "What requirements apply to outsourcing arrangements?",
    "What requirements relate to third-party service providers?",
    "What obligations apply to data management?",
    "What security controls are required?",
    "What requirements relate to incident management?",
    "What requirements apply to business continuity planning?",
    "What monitoring mechanisms are described?",
    "What supervisory expectations are outlined?",
    "What corrective actions are required in case of breach?",
    "What enforcement measures are mentioned?",
    "What review timelines are specified?",
    "What reporting lines are required?",
    "What overall regulatory expectations are stated?"
]


# ============================================================
# BATCH EVALUATION
# ============================================================

def main():

    log_name = f"graphrag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(os.getcwd(), log_name)
    sys_csv = log_name.replace(".txt", "_system.csv")

    monitor = SystemMonitor(sys_csv, interval=1.0)
    monitor.start()

    total_start = time.time()

    total_latency_list = []
    ttfb_list = []
    gen_list = []
    tps_list = []

    with open(log_path, "w", encoding="utf-8") as log:

        log.write("GraphRAG Batch Evaluation\n")
        log.write(f"Start Time: {datetime.now()}\n")
        log.write("="*90 + "\n\n")
        log.flush()

        for i, question in enumerate(QUESTIONS, 1):

            print(f"\n[{i}/{len(QUESTIONS)}] {question}")
            log.write(f"\nQUESTION {i}: {question}\n")

            q_start = time.time()

            # ---------- Retrieval ----------
            r_start = time.time()
            q_emb = get_embedding(question)
            hits = retrieve_top_k(q_emb, k=3)
            retrieval_time = time.time() - r_start

            best = hits[0]
            with driver.session() as session:
                expanded = expand_node(session, best["node"].element_id, best["label"])

            context = format_context(best["label"], expanded)

            # ---------- LLM ----------
            metrics = call_ollama_metrics(context, question)

            total_latency_list.append(metrics["total_latency"])
            ttfb_list.append(metrics["ttfb"])
            gen_list.append(metrics["gen_time"])
            tps_list.append(metrics["tps"])

            # ---------- Logging ----------
            log.write("ANSWER:\n" + metrics["answer"] + "\n\n")

            log.write(f"timestamp_sent        : {metrics['timestamp_sent']}\n")
            log.write(f"timestamp_first_byte : {metrics['timestamp_first_byte']}\n")
            log.write(f"timestamp_completed  : {metrics['timestamp_completed']}\n")

            log.write(f"ttfb_sec              : {metrics['ttfb']:.4f}\n")
            log.write(f"generation_time_sec  : {metrics['gen_time']:.4f}\n")
            log.write(f"total_latency_sec    : {metrics['total_latency']:.4f}\n")
            log.write(f"estimated_tokens     : {metrics['tokens']}\n")
            log.write(f"tokens_per_sec       : {metrics['tps']:.2f}\n")
            log.write(f"retrieval_time_sec   : {retrieval_time:.4f}\n")

            log.write("-"*90 + "\n")
            log.flush()

        total_time = time.time() - total_start
        monitor.stop()

        log.write("\n" + "="*90 + "\n")
        log.write("FINAL SUMMARY\n")
        log.write("="*90 + "\n")

        log.write(f"Total Test Duration (s): {total_time:.2f}\n")
        log.write(f"Total Requests        : {len(QUESTIONS)}\n")
        log.write(f"Successful Requests   : {len(QUESTIONS)}\n")
        log.write(f"Success Rate (%)      : 100\n\n")

        log.write(f"Avg TTFB (s)          : {sum(ttfb_list)/len(ttfb_list):.4f}\n")
        log.write(f"Avg Gen Time (s)      : {sum(gen_list)/len(gen_list):.4f}\n")
        log.write(f"Avg Total Latency (s) : {sum(total_latency_list)/len(total_latency_list):.4f}\n")
        log.write(f"Avg Tokens/sec        : {sum(tps_list)/len(tps_list):.2f}\n")
        log.write(f"Max Latency (s)       : {max(total_latency_list):.4f}\n")

    print("\n✅ GraphRAG benchmark completed.")
    print("📄 Log file:", log_path)
    print("📊 System metrics CSV:", sys_csv)


if __name__ == "__main__":
    main()
