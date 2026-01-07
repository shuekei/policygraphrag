import os
import json
import time
from datetime import datetime
from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, AutoModel
import requests

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

def call_ollama(context, question):

    prompt = f"""
You are a regulatory policy expert. Use ONLY the provided context.

The context includes file name, page number and clause hierarchy.
Preserve these when answering.

CONTEXT:
{context}

QUESTION:
{question}
"""

    payload = {
        "model": "llama3-chatqa:70b",
        "prompt": prompt,
        "stream": False
    }

    start = time.time()
    r = requests.post("http://localhost:11434/api/generate", json=payload)
    output = r.json()["response"]
    llm_time = time.time() - start

    return output, llm_time


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

    total_start = time.time()
    total_times = []

    with open(log_path, "w", encoding="utf-8") as log:

        log.write(f"GraphRAG Batch Evaluation\nStart: {datetime.now()}\n")
        log.write("="*80 + "\n\n")
        log.flush()

        for i, question in enumerate(QUESTIONS, 1):

            print(f"\n[{i}/{len(QUESTIONS)}] {question}")
            log.write(f"\nQUESTION {i}: {question}\n")

            q_start = time.time()

            retrieval_start = time.time()
            q_emb = get_embedding(question)
            hits = retrieve_top_k(q_emb, k=3)
            retrieval_time = time.time() - retrieval_start

            best = hits[0]
            with driver.session() as session:
                expanded = expand_node(session, best["node"].element_id, best["label"])

            context = format_context(best["label"], expanded)
            answer, llm_time = call_ollama(context, question)

            total_time = time.time() - q_start
            total_times.append(total_time)

            log.write("ANSWER:\n" + answer + "\n\n")
            log.write(f"Retrieval time : {retrieval_time:.2f}s\n")
            log.write(f"LLM time       : {llm_time:.2f}s\n")
            log.write(f"Total time     : {total_time:.2f}s\n")
            log.write("-"*80 + "\n")
            log.flush()

        avg_time = sum(total_times) / len(total_times)
        all_time = time.time() - total_start

        log.write("\n" + "="*80 + "\n")
        log.write(f"TOTAL QUESTIONS : {len(QUESTIONS)}\n")
        log.write(f"TOTAL TIME      : {all_time:.2f}s\n")
        log.write(f"AVERAGE TIME    : {avg_time:.2f}s\n")

    print("\n✅ Batch evaluation completed.")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
