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
import pynvml

if torch.cuda.is_available():
    pynvml.nvmlInit()
# ============================================================
# NEO4J CONNECTION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("✅ Neo4j connected.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("🤖 LLM model       :", "qwen3:30b")

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
                "gpu_util_percent","gpu_mem_util_percent",
                "gpu_mem_mib","gpu_temp_c","gpu_power_w"
            ])

            while self.running:
                ts = datetime.now().strftime("%H:%M:%S")
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                ram_pct = mem.percent
                ram_gb = mem.used / (1024**3)

                if torch.cuda.is_available():
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                        # ---- GPU core util ----
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_util = util.gpu
                            gpu_mem_util = util.memory
                        except pynvml.NVMLError:
                            gpu_util, gpu_mem_util = -1, -1
                
                        # ---- GPU memory used ----
                        try:
                            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_mem = meminfo.used / (1024**2)
                        except pynvml.NVMLError:
                            gpu_mem = -1
                
                        # ---- Temperature ----
                        try:
                            temp = pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        except pynvml.NVMLError:
                            temp = -1
                
                        # ---- Power ----
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                        except pynvml.NVMLError:
                            power = -1
                
                    except pynvml.NVMLError:
                        gpu_util, gpu_mem_util, gpu_mem, temp, power = -1, -1, -1, -1, -1
                else:
                    gpu_util, gpu_mem_util, gpu_mem, temp, power = 0, 0, 0, 0, 0


                writer.writerow([
                    ts,
                    round(cpu,2),
                    round(ram_pct,2),
                    round(ram_gb,2),
                    round(gpu_util,2),
                    round(gpu_mem_util,2),
                    round(gpu_mem,0),
                    temp,
                    round(power,2)
                ])

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
        "model": "qwen3:30b",
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
    # Capital & Liquidity
    "What is the minimum Leverage Ratio that banking institutions must maintain at all times?",
    "What is the objective of the Liquidity Coverage Ratio (LCR)?",
    "What is the cap on inflows that can offset outflows in the LCR calculation?",
    "What is the minimum Net Stable Funding Ratio (NSFR) requirement?",
    "What must a financial institution do if its NSFR falls below the 100% threshold?",
    "What is the standard rate for the Capital Conservation Buffer (CCB)?",
    "What is the minimum capital planning horizon for stress test reporting by banking institutions?",

    # Investments, Equity & Market Risk
    "What is the aggregate limit for a financial institution's non-financial investments?",
    "What defines 'material interest' in a corporation according to the equity investment guidelines?",
    "What is the maximum interest in shares a DFI member can have in a corporation before they are prohibited from receiving financing?",
    "What is the Look-through Approach (LTA) for equity investments in funds?",
    "What is the maximum short position an Eligible Market Participant can hold in a single security issue?",
    "How does the Bank define a Qualifying Central Counterparty (QCCP)?",
    "Which entities are considered core market participants for 0% risk weight exemptions in repo-style transactions?",

    # Operational Risk & Reporting
    "What are the components of the Business Indicator (BI) used to calculate operational risk capital?",
    "What are the three buckets of the Business Indicator used for operational risk coefficients?",
    "Who is responsible for ensuring the accuracy of regulatory reports submitted to the Bank?",
    "What is the reporting threshold for a customer information breach to be considered significant scale?",
    "How often must financial service providers conduct an independent review of their customer information protection policies?",
    "What must an FSP designate to oversee the implementation of customer information protection policies?",

    # Money Services Business (MSB)
    "What is the annual fee for a Money Services Business (MSB) branch office?",
    "What is the deadline for MSB licensees to pay their annual fees each year?",
    "What is the minimum attendance requirement for directors of medium and large MSBs?",

    # Insurance & Takaful
    "How quickly must an insurer pay out a motor death claim?",
    "What is the minimum free look period for a family takaful product?",
    "What is the required turnaround time for a CITO to respond to a referral for a TPPD claim?",
    "What is the scale of betterment in motor claims?",
    "What is the maximum rate of betterment for a vehicle aged 10 years and above?",

    # Islamic Finance & Shariah
    "How are losses distributed in a Mudarabah contract?",
    "How is profit defined in a Mudarabah contract?",
    "What defines a Musyarakah Mutanaqisah contract?",
    "What is the repayment requirement for a borrower under a Qard contract?",
    "Under what conditions is an Istisna contract considered completed?",
    "What is the definition of constructive possession in an Istisna contract?",
    "What is the reporting timeline for a Shariah Non-Compliance (SNC) event confirmation?",

    # Sukuk & Project Finance
    "What is the difference between asset-based and asset-backed Sukuk?",
    "What are the eligibility criteria for high-quality project finance exposures during the operational phase?",

    # Climate Risk & Sustainability
    "What are the three drivers of climate-related risks?",
    "What are the three scopes of Greenhouse Gas (GHG) emissions?",
    "What are the Stretch recommendations for TCFD climate disclosures?",

    # IT, Security & Systems
    "What security infrastructure provides data authenticity and non-repudiation for RENTAS transactions?",
    "Which architectural style is recommended for financial institutions publishing Open Data APIs?",
    "What is the primary purpose of the ORION system?",
    "How quickly must a robbery involving more than RM200,000 be reported in the ORION system?",
    "How long must records of transaction communication for wholesale dealings be retained?",

    # Governance & Licensing
    "What is the cooling-off period for an external auditor before resuming the role of engagement partner for a DFI?",
    "What constitutes a non-complex case for the release of a property's original title?",
    "What determines the best interest of Malaysia in a new license application?",

    # Consumer Protection
    "What are the five key principles of the Perlindungan Tenang initiative?"
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



















