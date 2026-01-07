import os
import json
import time
from datetime import datetime
from pathlib import Path
import psutil
import threading
from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# PATH SETUP (portable)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "test_data"
CLEANED_DIR = DATA_DIR / "cleaned"
LOG_DIR = DATA_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / f"neo4j_insert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

json_files = list(CLEANED_DIR.glob("*.json"))

print(f"Found {len(json_files)} cleaned files.")


# ============================================================
# NEO4J CONNECTION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("✅ Neo4j connection established.")


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
# VECTOR INDEX CREATION
# ============================================================

def create_vector_indexes(driver):

    queries = [
        """
        CREATE VECTOR INDEX clause_embedding_index IF NOT EXISTS
        FOR (c:Clause)
        ON c.embedding
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: "cosine"
          }
        }
        """,

        """
        CREATE VECTOR INDEX section_embedding_index IF NOT EXISTS
        FOR (s:Section)
        ON s.embedding
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: "cosine"
          }
        }
        """,

        """
        CREATE VECTOR INDEX subsection_embedding_index IF NOT EXISTS
        FOR (ss:Subsection)
        ON ss.embedding
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: "cosine"
          }
        }
        """,

        """
        CREATE VECTOR INDEX part_embedding_index IF NOT EXISTS
        FOR (p:Part)
        ON p.embedding
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: "cosine"
          }
        }
        """
    ]

    with driver.session() as session:
        for q in queries:
            session.run(q)

    print("✅ Vector indexes are created / already exist.")



# ============================================================
# NEO4J HELPERS
# ============================================================

def create_node(tx, label, props):
    query = f"""
    CREATE (n:{label} $props)
    RETURN elementId(n) AS node_id
    """
    res = tx.run(query, props=props)
    return res.single()["node_id"]



def create_rel(tx, from_id, to_id, rel):
    query = f"""
    MATCH (a), (b)
    WHERE elementId(a) = $from_id AND elementId(b) = $to_id
    CREATE (a)-[:`{rel}`]->(b)
    """
    tx.run(query, from_id=from_id, to_id=to_id)

# ============================================================
# SYSTEM MONITOR
# ============================================================

class SystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu = []
        self.ram = []
        self.gpu = []
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _run(self):
        process = psutil.Process(os.getpid())

        while self._running:
            # CPU %
            self.cpu.append(psutil.cpu_percent(interval=None))

            # RAM %
            self.ram.append(psutil.virtual_memory().percent)

            # GPU % (if available)
            if torch.cuda.is_available():
                try:
                    util = torch.cuda.utilization()
                except:
                    util = None

                if util is None:
                    # fallback: memory usage ratio
                    used = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    self.gpu.append((used / total) * 100)
                else:
                    self.gpu.append(util)
            else:
                self.gpu.append(0)

            time.sleep(self.interval)

    def summary(self):
        return {
            "avg_cpu": sum(self.cpu)/len(self.cpu) if self.cpu else 0,
            "max_cpu": max(self.cpu) if self.cpu else 0,
            "avg_ram": sum(self.ram)/len(self.ram) if self.ram else 0,
            "max_ram": max(self.ram) if self.ram else 0,
            "avg_gpu": sum(self.gpu)/len(self.gpu) if self.gpu else 0,
            "max_gpu": max(self.gpu) if self.gpu else 0,
        }

# ============================================================
# MAIN
# ============================================================

def main():

    if not json_files:
        print("❌ No cleaned JSON files found.")
        return

    print("\n🔧 Creating vector indexes (if not exist)...")
    create_vector_indexes(driver)

    total_requests = len(json_files)
    success_requests = 0
    latencies = []

    print("\n📊 Starting system monitoring...")
    monitor = SystemMonitor(interval=1.0)
    monitor.start()

    total_start = time.time()

    with open(LOG_PATH, "w", encoding="utf-8") as log:

        log.write(f"Neo4j KG Ingestion Log\nStart Time: {datetime.now()}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        with driver.session() as session:

            for idx, json_path in enumerate(json_files, 1):

                print(f"\n[{idx}/{len(json_files)}] Processing: {json_path.name}")
                file_start = time.time()

                try:
                    # -----------------------------
                    # LOAD ONE CLEANED FILE
                    # -----------------------------
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    source_file_name = data.get("source_file", json_path.stem)
                    parts = data.get("parts", [])

                    # -----------------------------
                    # CREATE SOURCE FILE NODE
                    # -----------------------------
                    source_emb = get_embedding(source_file_name)

                    source_id = session.execute_write(
                        create_node,
                        "SourceFile",
                        {
                            "file_name": source_file_name,
                            "text": source_file_name,
                            "embedding": source_emb
                        }
                    )

                    # ----------------------------
                    # INSERT GRAPH STRUCTURE
                    # ----------------------------
                    for part in parts:
                        part_title = part["title"]
                        part_emb = get_embedding(part_title)

                        part_id = session.execute_write(
                            create_node,
                            "Part",
                            {
                                "title": part_title,
                                "page_no": part.get("page_no"),
                                "text": part_title,
                                "embedding": part_emb,
                                "source_file": source_file_name,
                                "part_title": part_title
                            }
                        )

                        session.execute_write(create_rel, source_id, part_id, "HAS_PART")

                        for section in part["sections"]:
                            section_title = section["title"]
                            section_emb = get_embedding(section_title)

                            section_id = session.execute_write(
                                create_node,
                                "Section",
                                {
                                    "title": section_title,
                                    "page_no": section.get("page_no"),
                                    "text": section_title,
                                    "embedding": section_emb,
                                    "source_file": source_file_name,
                                    "part_title": part_title,
                                    "section_title": section_title
                                }
                            )

                            session.execute_write(create_rel, part_id, section_id, "HAS_SECTION")

                            # ---------- SUBSECTIONS ----------
                            for subsection in section.get("subsections", []):
                                sub_title = subsection["title"]
                                sub_emb = get_embedding(sub_title)

                                sub_id = session.execute_write(
                                    create_node,
                                    "Subsection",
                                    {
                                        "title": sub_title,
                                        "page_no": subsection.get("page_no"),
                                        "text": sub_title,
                                        "embedding": sub_emb,
                                        "source_file": source_file_name,
                                        "part_title": part_title,
                                        "section_title": section_title,
                                        "subsection_title": sub_title
                                    }
                                )

                                session.execute_write(create_rel, section_id, sub_id, "HAS_SUBSECTION")

                                for clause in subsection.get("clauses", []):
                                    clause_text = clause["text"]
                                    clause_emb = get_embedding(clause_text)

                                    clause_id = session.execute_write(
                                        create_node,
                                        "Clause",
                                        {
                                            "clause_number": clause.get("clause_number", ""),
                                            "page_no": clause.get("page_no"),
                                            "text": clause_text,
                                            "embedding": clause_emb,
                                            "source_file": source_file_name,
                                            "part_title": part_title,
                                            "section_title": section_title,
                                            "subsection_title": sub_title
                                        }
                                    )

                                    session.execute_write(create_rel, sub_id, clause_id, "HAS_CLAUSE")

                            # ---------- SECTION CLAUSES ----------
                            for clause in section.get("clauses", []):
                                clause_text = clause["text"]
                                clause_emb = get_embedding(clause_text)

                                clause_id = session.execute_write(
                                    create_node,
                                    "Clause",
                                    {
                                        "clause_number": clause.get("clause_number", ""),
                                        "page_no": clause.get("page_no"),
                                        "text": clause_text,
                                        "embedding": clause_emb,
                                        "source_file": source_file_name,
                                        "part_title": part_title,
                                        "section_title": section_title
                                    }
                                )

                                session.execute_write(create_rel, section_id, clause_id, "HAS_CLAUSE")

                    success_requests += 1

                except Exception as e:
                    print("❌ Failed:", json_path.name, str(e))
                    log.write(f"{json_path.name} -> FAILED: {e}\n")
                    log.flush()
                    continue

                # -----------------------------
                # PER-FILE TIMING
                # -----------------------------
                file_time = time.time() - file_start
                latencies.append(file_time)

                msg = f"{json_path.name} -> {file_time:.2f} seconds"
                print("   ", msg)
                log.write(msg + "\n")
                log.flush()

        # -----------------------------
        # STOP MONITOR & SUMMARY
        # -----------------------------
        monitor.stop()
        sys_stats = monitor.summary()

        total_time = time.time() - total_start
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        success_rate = (success_requests / total_requests) * 100

        log.write("\n" + "=" * 80 + "\n")
        log.write("SYSTEM PERFORMANCE SUMMARY\n")
        log.write("=" * 80 + "\n")

        log.write(f"Total Test Duration (s) : {total_time:.2f}\n")
        log.write(f"Total Requests         : {total_requests}\n")
        log.write(f"Successful Requests    : {success_requests}\n")
        log.write(f"Success Rate (%)       : {success_rate:.2f}\n\n")

        log.write(f"Avg Latency (s)        : {avg_latency:.2f}\n")
        log.write(f"Max Latency (s)        : {max_latency:.2f}\n\n")

        log.write(f"Avg CPU (%)            : {sys_stats['avg_cpu']:.2f}\n")
        log.write(f"Max CPU (%)            : {sys_stats['max_cpu']:.2f}\n")
        log.write(f"Avg RAM (%)            : {sys_stats['avg_ram']:.2f}\n")
        log.write(f"Max RAM (%)            : {sys_stats['max_ram']:.2f}\n")
        log.write(f"Avg GPU (%)            : {sys_stats['avg_gpu']:.2f}\n")
        log.write(f"Max GPU (%)            : {sys_stats['max_gpu']:.2f}\n")

        log.flush()

    print("\n✅ All Neo4j insertions completed.")
    print(f"⏱ Total time   : {total_time:.2f} seconds")
    print(f"📊 Avg latency : {avg_latency:.2f} seconds/file")
    print(f"📝 Log saved to: {LOG_PATH}")



# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()

