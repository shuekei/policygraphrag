import os
import json
import time
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
print("✅ Connected to Neo4j.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to(device)

def get_embedding(text: str):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=8192
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)

    if hasattr(out, "embeddings") and out.embeddings is not None:
        return out.embeddings.squeeze(0).cpu().numpy().tolist()

    hidden = out.last_hidden_state
    emb = hidden.mean(dim=1)
    return emb.squeeze(0).cpu().numpy().tolist()


# ============================================================
# NEO4J VECTOR SEARCH
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
# INTERACTIVE CLI
# ============================================================

def graphrag_query(question, k=3):

    total_start = time.time()

    retrieval_start = time.time()
    q_emb = get_embedding(question)
    top_hits = retrieve_top_k(q_emb, k)
    retrieval_time = time.time() - retrieval_start

    best = top_hits[0]
    best_node = best["node"]
    best_label = best["label"]

    with driver.session() as session:
        expanded = expand_node(session, best_node.element_id, best_label)

    context = format_context(best_label, expanded)
    answer, llm_time = call_ollama(context, question)
    total_time = time.time() - total_start

    print("\n==============================")
    print("🤖 ANSWER")
    print("==============================\n")
    print(answer)

    print("\n==============================")
    print("⏱ TIMING")
    print("==============================")
    print(f"Retrieval : {retrieval_time:.2f}s")
    print(f"LLM       : {llm_time:.2f}s")
    print(f"Total     : {total_time:.2f}s\n")


def main():
    print("\n==============================")
    print("🚀 GraphRAG CLI (Ollama)")
    print("Type 'exit' or 'quit' to stop")
    print("==============================\n")

    while True:
        try:
            question = input("🧠 Ask > ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            if not question:
                continue
            graphrag_query(question)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
