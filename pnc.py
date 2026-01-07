import os
import json
import re
import time
from datetime import datetime
from docling.document_converter import DocumentConverter

# ============================================================
# PATH SETUP (portable)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "test_data")
RAW_DIR = os.path.join(DATA_DIR, "raw_docling")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
LOG_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    LOG_DIR, f"parsing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# ============================================================
# HELPERS
# ============================================================

def get_page_no(node):
    prov = node.get("prov") or []
    if prov and isinstance(prov, list):
        return prov[0].get("page_no")
    return None

def clean_text(txt):
    return " ".join((txt or "").split())

RE_PART     = re.compile(r"^PART\s+[A-Z]", re.IGNORECASE)
RE_SECTION  = re.compile(r"^\d+\s+")
RE_CLAUSE   = re.compile(r"^(?:[SG]\s*)?(?P<num>\d+\.\d+)\s+(?P<rest>.+)")
RE_SUBPOINT = re.compile(r"^\((?P<key>[a-z0-9]+)\)\s+(?P<rest>.+)", re.IGNORECASE)

# ============================================================
# PARSER
# ============================================================

def parse_docling_json(raw_doc):

    source_file = None
    if "origin" in raw_doc and isinstance(raw_doc["origin"], dict):
        source_file = raw_doc["origin"].get("filename")

    texts   = {t["self_ref"]: t for t in raw_doc.get("texts", [])}
    groups  = {g["self_ref"]: g for g in raw_doc.get("groups", [])}
    tables  = {tb["self_ref"]: tb for tb in raw_doc.get("tables", [])}
    body_children = raw_doc["body"]["children"]

    document = {
        "source_file": source_file,
        "parts": [],
        "footnotes": []
    }

    current_part = None
    current_section = None
    current_subsection = None
    current_clause = None

    def ensure_part():
        nonlocal current_part
        if current_part is None:
            current_part = {
                "type": "part",
                "title": "PART 0 PREAMBLE",
                "page_no": None,
                "sections": []
            }
            document["parts"].append(current_part)

    def ensure_section(page_no):
        nonlocal current_section
        ensure_part()
        if current_section is None:
            current_section = {
                "type": "section",
                "title": "Untitled Section",
                "page_no": page_no,
                "subsections": [],
                "clauses": []
            }
            current_part["sections"].append(current_section)

    def walk_children(children):
        for ch in children:
            cref = ch.get("cref")
            if not cref:
                continue
            if cref in texts:
                handle_text(texts[cref])
            elif cref in groups:
                walk_children(groups[cref].get("children", []))
            elif cref in tables:
                continue

    def handle_text(t):
        nonlocal current_part, current_section, current_subsection, current_clause

        label = t.get("label")
        text  = clean_text(t.get("text", ""))
        page_no = get_page_no(t)

        if not text:
            return

        if label in ("page_header", "page_footer"):
            return

        if label == "footnote":
            document["footnotes"].append({"text": text, "page_no": page_no})
            return

        if RE_PART.match(text):
            current_part = {"type": "part","title": text,"page_no": page_no,"sections": []}
            document["parts"].append(current_part)
            current_section = current_subsection = current_clause = None
            return

        if RE_SECTION.match(text):
            ensure_part()
            current_section = {"type": "section","title": text,"page_no": page_no,"subsections": [],"clauses": []}
            current_part["sections"].append(current_section)
            current_subsection = current_clause = None
            return

        m_clause = RE_CLAUSE.match(text)
        if m_clause:
            ensure_section(page_no)
            target = current_subsection or current_section
            current_clause = {
                "type": "clause",
                "clause_number": m_clause.group("num"),
                "page_no": page_no,
                "text": m_clause.group("rest").strip()
            }
            target["clauses"].append(current_clause)
            return

        m_sub = RE_SUBPOINT.match(text)
        if m_sub and current_clause:
            current_clause["text"] += f" ({m_sub.group('key')}) {m_sub.group('rest').strip()}"
            return

        if current_clause:
            current_clause["text"] += " " + text
            return

    walk_children(body_children)
    return document

# ============================================================
# MAIN
# ============================================================

def main():

    converter = DocumentConverter()
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in:", DATA_DIR)
        return

    print(f"Found {len(pdf_files)} PDFs")

    total_start = time.time()
    file_times = []

    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"Parsing log started at {datetime.now()}\n")
        log.write("="*60 + "\n\n")
        log.flush()

        for idx, pdf_name in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_name}")

            file_start = time.time()
            pdf_path = os.path.join(DATA_DIR, pdf_name)
            base = os.path.splitext(pdf_name)[0]

            # ---- Convert ----
            result = converter.convert(pdf_path)
            doc = result.document
            raw_dict = doc.model_dump()

            raw_path = os.path.join(RAW_DIR, base + "_raw.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_dict, f, ensure_ascii=False, indent=2)

            # ---- Parse ----
            structured_doc = parse_docling_json(raw_dict)

            clean_path = os.path.join(CLEAN_DIR, base + "_cleaned.json")
            with open(clean_path, "w", encoding="utf-8") as f:
                json.dump(structured_doc, f, ensure_ascii=False, indent=2)

            file_time = time.time() - file_start
            file_times.append(file_time)

            msg = f"{pdf_name} -> {file_time:.2f} seconds"
            print("   ", msg)

            # ✅ write immediately
            log.write(msg + "\n")
            log.flush()

        total_time = time.time() - total_start
        avg_time = sum(file_times) / len(file_times)

        log.write("\n" + "="*60 + "\n")
        log.write(f"Total files   : {len(file_times)}\n")
        log.write(f"Total time    : {total_time:.2f} seconds\n")
        log.write(f"Average time  : {avg_time:.2f} seconds per file\n")
        log.flush()

    print("\n✅ All files processed.")
    print(f"⏱ Total time   : {total_time:.2f} seconds")
    print(f"📊 Average time: {avg_time:.2f} seconds/file")
    print(f"📝 Log saved to: {LOG_PATH}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
