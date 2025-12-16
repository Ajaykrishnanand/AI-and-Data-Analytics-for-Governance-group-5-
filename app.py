# streamlit_app.py
import streamlit as st
import os, io, json, re, math, tempfile, uuid, hashlib
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'all_collections' not in st.session_state:
    st.session_state.all_collections = {}

# --------- CONFIG ---------
@st.cache_resource
def load_config():
    config = {
        # Update these paths for your system
        "POPLER_PATH": r"C:\Users\lenovo\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
        "TESSERACT_CMD": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "EMBED_MODEL_NAME": "sentence-transformers/paraphrase-mpnet-base-v2",
        "EMBED_BATCH": 64,
        "CLASSIFIER_MODEL": "llama3:8b",
        "EXTRACTION_MODEL": "llama3:8b",
        "CHROMA_DIR": "chroma_store",
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 200
    }
    
    try:
        pytesseract.pytesseract.tesseract_cmd = config["TESSERACT_CMD"]
    except:
        st.warning("Tesseract OCR path might be incorrect. OCR functionality may not work properly.")
    
    return config

config = load_config()

# Load embedding model
@st.cache_resource
def load_embed_model():
    try:
        model = SentenceTransformer(config["EMBED_MODEL_NAME"])
        st.sidebar.success("✅ Embedding model loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load embedding model: {e}")
        return None

embed_model = load_embed_model()

# Chroma client
@st.cache_resource
def get_chroma_client():
    try:
        client = chromadb.PersistentClient(path=config["CHROMA_DIR"])
        st.sidebar.success("✅ ChromaDB connected")
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to connect to ChromaDB: {e}")
        return None

# Helper function to generate unique collection name for each PDF
def generate_collection_name(pdf_path: str, file_content: bytes = None) -> str:
    """Generate a unique collection name based on file content/hash."""
    if file_content:
        # Generate hash from file content
        file_hash = hashlib.md5(file_content).hexdigest()[:12]
        filename = os.path.basename(pdf_path)
        safe_name = re.sub(r'[^\w\-_]', '_', filename[:50])
        return f"doc_{safe_name}_{file_hash}"
    else:
        # Generate hash from file path and size
        file_stat = os.stat(pdf_path)
        unique_str = f"{pdf_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        file_hash = hashlib.md5(unique_str.encode()).hexdigest()[:12]
        filename = os.path.basename(pdf_path)
        safe_name = re.sub(r'[^\w\-_]', '_', filename[:50])
        return f"doc_{safe_name}_{file_hash}"

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content."""
    return hashlib.md5(file_content).hexdigest()[:12]

# Text extraction functions
def extract_text_auto(pdf_path: str, ocr_dpi=300, ocr_lang="eng") -> List[str]:
    """Extract text from PDF with OCR fallback."""
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(pdf_path)
        pages_text = []
        
        progress_bar = st.progress(0)
        for i, page in enumerate(doc, start=1):
            try:
                page_text = page.get_text().strip()
            except Exception:
                page_text = ""
            
            if page_text and len(page_text) > 40:
                pages_text.append(page_text)
            else:
                try:
                    pix = page.get_pixmap(dpi=ocr_dpi)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                except Exception:
                    images = convert_from_path(pdf_path, dpi=ocr_dpi, 
                                             poppler_path=config["POPLER_PATH"])
                    img = images[i-1] if i-1 < len(images) else images[0]
                
                ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
                pages_text.append(ocr_text)
            
            progress_bar.progress(i / len(doc))
        
        progress_bar.empty()
        return pages_text

def join_pages(pages: List[str]) -> str:
    """Join page texts into a single string."""
    txt = "\n".join(pages)
    return " ".join(txt.split())

def chunk_text(text: str, chunk_size=config["CHUNK_SIZE"], 
               overlap=config["CHUNK_OVERLAP"]) -> List[str]:
    """Chunk text with overlap."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

# Extraction prompts (same as your file)
def get_extraction_prompt(category, text):
    # ============================================================
    # 1) CLAIM DOCUMENT
    # ============================================================
    if category == "Claim Document":
        return f"""
You are an expert in extracting structured fields from HOSPITAL INSURANCE CLAIM FORMS.
Extract ONLY values explicitly visible — NO hallucination.

CRITICAL RULES:
1. Missing → return "Not mentioned".
2. Preserve original formatting of dates, amounts, and labels.
3. Checkboxes: ✔ / ✓ / X / Yes / Selected = mark as present.
4. Keep OCR errors minimal — extract only clear text.
5. AI Summary must be max 2 sentences.

FIELDS TO EXTRACT:
- Primary Insured Name
- Policy Number
- TPA / Company ID
- Employee / Member ID
- Patient Name
- Insurance Company Name
- Hospital Name
- Hospital Type
- Admission Date
- Discharge Date
- Injury / Illness Type
- Claim Type
- Claim Amount
- Billing Breakdown:
    - Pre Hospitalization
    - Hospitalization
    - Post Hospitalization
    - Pharmacy Bills
    - Ambulance Charges
    - Other Charges
- Submitted Documents Checklist (list)
- City / Location
- AI Summary

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
  "Primary Insured Name": "",
  "Policy Number": "",
  "TPA / Company ID": "",
  "Employee / Member ID": "",
  "Patient Name": "",
  "Insurance Company Name": "",
  "Hospital Name": "",
  "Hospital Type": "",
  "Admission Date": "",
  "Discharge Date": "",
  "Injury / Illness Type": "",
  "Claim Type": "",
  "Claim Amount": "",
  "Billing Breakdown": {{
    "Pre Hospitalization": "",
    "Hospitalization": "",
    "Post Hospitalization": "",
    "Pharmacy Bills": "",
    "Ambulance Charges": "",
    "Other Charges": ""
  }},
  "Submitted Documents Checklist": [],
  "City / Location": "",
  "AI Summary": ""
}}
"""

    # ============================================================
    # 2) HEALTH INSURANCE POLICY
    # ============================================================
    if category == "Health Insurance Policy":
        return f"""
You are an expert in HEALTH INSURANCE POLICY extraction.

RULES:
- Extract ONLY explicit text from document.
- Missing → "Not mentioned".
- Preserve formatting.

FIELDS:
- Policy Holder Name
- Policy Number
- Insurance Company
- TPA Name
- Sum Insured
- Coverage Type
- Policy Start Date
- Policy End Date
- UIN / Product Code
- AI Summary (max 2 sentences)

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Policy Holder Name": "",
 "Policy Number": "",
 "Insurance Company": "",
 "TPA Name": "",
 "Sum Insured": "",
 "Coverage Type": "",
 "Policy Start Date": "",
 "Policy End Date": "",
 "UIN / Product Code": "",
 "AI Summary": ""
}}
"""

    # ============================================================
    # 3) MOTOR INSURANCE POLICY
    # ============================================================
    if category == "Motor Insurance Policy":
        return f"""
You extract structured fields from MOTOR INSURANCE POLICY documents.

RULES:
- No hallucination.
- Preserve original formatting.

FIELDS:
- Policy Holder Name
- Policy Number
- Insurance Company
- Coverage Type
- Vehicle Model
- Registration Number
- Engine / Chassis Number
- Policy Start Date
- Policy End Date
- IDV (if present)
- UIN
- AI Summary

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Policy Holder Name": "",
 "Policy Number": "",
 "Insurance Company": "",
 "Coverage Type": "",
 "Vehicle Model": "",
 "Registration Number": "",
 "Engine / Chassis Number": "",
 "Policy Start Date": "",
 "Policy End Date": "",
 "IDV": "",
 "UIN": "",
 "AI Summary": ""
}}
"""

    # ============================================================
    # 4) LIFE INSURANCE POLICY
    # ============================================================
    if category == "Life Insurance Policy":
        return f"""
You are an expert in extracting structured fields from LIFE INSURANCE and PERSONAL ACCIDENT INSURANCE CERTIFICATES.

RULES:
- Extract ONLY what is explicitly present in the document.
- If a field is missing → return "Not mentioned".
- Preserve formatting of dates, amounts, and numeric codes.
- Do NOT hallucinate.
- AI Summary = exactly 2 sentences.

FIELDS TO EXTRACT:
- Policy Holder Name
- Father's Name
- Address
- Certificate Number
- Intermediary Code
- Intermediary Name
- Policy Start Date
- Policy End Date
- Policy Duration
- Sum Assured / Sum Insured
- SBI Account Number
- Premium Amount
- Nominee Name
- Nominee Relationship
- Insurance Company
- UIN / Product Code
- Coverage List (list)
- Exclusions (list)
- Customer Care Contacts
- AI Summary

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Policy Holder Name": "",
 "Father's Name": "",
 "Address": "",
 "Certificate Number": "",
 "Intermediary Code": "",
 "Intermediary Name": "",
 "Policy Start Date": "",
 "Policy End Date": "",
 "Policy Duration": "",
 "Sum Assured / Sum Insured": "",
 "SBI Account Number": "",
 "Premium Amount": "",
 "Nominee Name": "",
 "Nominee Relationship": "",
 "Insurance Company": "",
 "UIN / Product Code": "",
 "Coverage List": [],
 "Exclusions": [],
 "Customer Care Contacts": "",
 "AI Summary": ""
}}
"""

    # ============================================================
    # 5) IRCTC TICKET
    # ============================================================
    if category == "IRCTC Ticket":
        return f"""
You extract structured fields from IRCTC train tickets.

FIELDS:
- Passenger Name
- Train Number
- Train Name
- Date of Journey
- Boarding Station
- Destination Station
- Class
- PNR
- Booking Status
- Current Status
- Coach / Seat
- Fare
- AI Summary

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Passenger Name": "",
 "Train Number": "",
 "Train Name": "",
 "Date of Journey": "",
 "Boarding Station": "",
 "Destination Station": "",
 "Class": "",
 "PNR": "",
 "Booking Status": "",
 "Current Status": "",
 "Coach / Seat": "",
 "Fare": "",
 "AI Summary": ""
}}
"""

    # ============================================================
    # 6) INVOICE
    # ============================================================
    if category == "Invoice":
        return f"""
You extract structured fields from INVOICE documents.

FIELDS:
- Invoice Number
- Invoice Date
- Vendor Name
- Customer Name
- Items (name, quantity, price, amount)
- Tax Amount
- Total Amount
- Payment Terms
- AI Summary

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Invoice Number": "",
 "Invoice Date": "",
 "Vendor Name": "",
 "Customer Name": "",
 "Items": [],
 "Tax Amount": "",
 "Total Amount": "",
 "Payment Terms": "",
 "AI Summary": ""
}}
"""

    # ============================================================
    # 7) LEGAL NOTICE
    # ============================================================
    if category == "Legal Notice":
        return f"""
You are an expert in extracting structured information from LEGAL NOTICE DOCUMENTS.

TASK:
Extract ONLY the fields explicitly present in the document.
If a field is missing → return "Not mentioned".
Never guess. Never add commentary.
Return ONLY valid JSON. No extra words.

STRICT RULES:
- Output MUST be EXACTLY a JSON dictionary.
- NO text before or after the JSON.
- NO markdown (no ```).
- Preserve original legal wording.
- AI Summary MUST be exactly 2 sentences.

FIELDS:
- Document Title
- Client / Complainant
- Accused / Respondent
- Advocate
- Issue Description
- Contract Amount
- Amount Pending
- Payment Deadline
- Interest Rate
- Notice Date
- Legal Actions Mentioned
- AI Summary

DOCUMENT:
{text}

NOW RETURN THIS JSON ONLY:
{{
  "Document Title": "",
  "Client / Complainant": "",
  "Accused / Respondent": "",
  "Advocate": "",
  "Issue Description": "",
  "Contract Amount": "",
  "Amount Pending": "",
  "Payment Deadline": "",
  "Interest Rate": "",
  "Notice Date": "",
  "Legal Actions Mentioned": "",
  "AI Summary": "",
}}
"""

    # ============================================================
    # FALLBACK
    # ============================================================
    return f"""
Extract ONLY explicit metadata from this document.

DOCUMENT:
{text}

Return STRICT JSON ONLY:
{{
 "Document Type": "",
 "Key Fields": "",
 "AI Summary": ""
}}
"""

# Chroma operations
def upsert_pdf_to_chroma(pdf_path: str, collection_name: str, clear_existing=False):
    """Upsert PDF chunks to Chroma."""
    with st.spinner("Processing PDF for vector database..."):
        pages = extract_text_auto(pdf_path)
        full_text = join_pages(pages)
        chunks = chunk_text(full_text)
        
        client = get_chroma_client()
        if client is None:
            st.error("ChromaDB client not available")
            return None
        
        try:
            col = client.get_collection(collection_name)
            if clear_existing:
                existing = col.get(include=["ids", "metadatas"])
                if existing.get("ids"):
                    col.delete(ids=existing["ids"])
        except Exception:
            col = client.create_collection(collection_name)
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": os.path.basename(pdf_path), "chunk_index": i, 
                      "collection": collection_name} for i in range(len(chunks))]
        
        # Compute embeddings
        embeddings = []
        progress_bar = st.progress(0)
        for i in range(0, len(chunks), config["EMBED_BATCH"]):
            batch = chunks[i:i+config["EMBED_BATCH"]]
            embs = embed_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            for row in embs:
                embeddings.append(row.tolist() if hasattr(row, "tolist") else list(row))
            progress_bar.progress(min((i + config["EMBED_BATCH"]) / len(chunks), 1.0))
        
        progress_bar.empty()
        
        if len(embeddings) != len(chunks):
            st.error(f"Embeddings length mismatch: {len(embeddings)} vs chunks {len(chunks)}")
            return None
        
        # Add to collection
        col.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        st.success(f"✅ Inserted {len(chunks)} chunks into collection '{collection_name}'")
        
        # Store in session state
        if collection_name not in st.session_state.all_collections:
            st.session_state.all_collections[collection_name] = {
                "filename": os.path.basename(pdf_path),
                "chunks": len(chunks),
                "created_at": pd.Timestamp.now()
            }
        
        return col

def retrieve_similar_chunks(query: str, collection_name: str, top_k=5):
    """Retrieve similar chunks from Chroma."""
    client = get_chroma_client()
    if client is None:
        return []
    
    try:
        col = client.get_collection(collection_name)
    except Exception:
        st.warning(f"Collection '{collection_name}' not found")
        return []
    
    # Compute query embedding
    q_emb = embed_model.encode([query])[0]
    try:
        q_emb = q_emb.tolist()
    except:
        pass
    
    # Query Chroma
    results = col.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)
    
    safe_docs = [(doc if isinstance(doc, str) else "") for doc in docs]
    return list(zip(safe_docs, metas, distances))

# Category classification
CATEGORY_KEYWORDS = {
    "Claim Document": [
        "claim", "admission", "discharge", "diagnosis", "tpa",
        "claim amount", "hospital", "pre-authorization"
    ],
    "Health Insurance Policy": [
        "policy", "sum insured", "uin", "health insurance",
        "coverage", "policy schedule"
    ],
    "Motor Insurance Policy": [
        "motor", "vehicle", "engine", "chassis", "registration",
        "idv", "two wheeler", "four wheeler"
    ],
    "Life Insurance Policy": [
        "life assured", "sum assured", "premium", "maturity",
        "death benefit", "survival benefit", "policy term"
    ],
    "Hospital Bill": [
        "bill", "invoice", "hospital charges", "room rent",
        "doctor fee", "pharmacy", "investigation"
    ],
    "Payment Receipt": [
        "receipt", "amount paid", "paid on", "transaction id",
        "payment received", "cash received"
    ],
    "KYC / Identity Document": [
        "passport", "aadhar", "aadhaar", "identity", "dob",
        "pan", "voter id", "driving license"
    ],
    "IRCTC Ticket": [
        "pnr", "train", "railway", "departure", "arrival",
        "berth", "coach", "irctc", "journey"
    ],
    "Invoice": [
        "invoice", "gst", "total payable", "unit price", "quantity",
        "tax invoice", "hsn", "igst", "cgst", "sgst"
    ],
    "Legal Notice": [
        "legal notice", "advocate", "lawyer", "contractor",
        "agreement", "pending amount", "breach", "serve notice",
        "defaulter", "obligation", "due amount", "legal action"
    ]
}

def embedding_candidate_category(chunks_with_meta):
    """Quick category classification using keyword matching."""
    scores = {}
    
    for doc, meta, dist in chunks_with_meta:
        if not doc:
            continue
        
        snippet = str(doc).lower()
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for k in keywords:
                if k in snippet:
                    scores[category] = scores.get(category, 0) + 1
    
    if not scores:
        return []
    
    valid_chunks = max(1, sum(1 for doc, _, _ in chunks_with_meta if doc))
    
    for c in scores:
        scores[c] /= valid_chunks
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def llm_classify_with_context(text: str, model=config["CLASSIFIER_MODEL"], top_k_snippets: List[str] = None):
    """Classify document using LLM."""
    ctx = "\n\n".join(top_k_snippets) if top_k_snippets else text[:3000]
    
    prompt = f"""
You are a highly accurate professional document classifier.
Classify the document into exactly ONE category from this list:

- Claim Document
- Health Insurance Policy
- Motor Insurance Policy
- Life Insurance Policy
- Hospital Bill
- Payment Receipt
- KYC / Identity Document
- IRCTC Ticket
- Invoice
- Legal Notice
- Other

RULES:
1. Use ONLY explicit text visible in the context.
2. If unsure or ambiguous → return "Other".
3. Output ONLY valid JSON.
4. Confidence MUST be between 0 and 1.

Return JSON only in this exact format:
{{ "category": "", "confidence": 0.0 }}

CONTEXT:
{ctx}
"""
    
    try:
        resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        content = resp["message"]["content"].strip()
    except Exception as e:
        st.error(f"LLM classifier failed: {e}")
        return {"category": "Other", "confidence": 0.0}
    
    # Parse JSON
    try:
        parsed = json.loads(content)
        if "category" in parsed and "confidence" in parsed:
            return parsed
    except:
        pass
    
    m = re.search(r"\{[\s\S]*?\}", content)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "category" in obj and "confidence" in obj:
                return obj
        except:
            pass
    
    return {"category": "Other", "confidence": 0.3}

# Field extraction
def extract_fields_via_llm(text: str, category: str, model=config["EXTRACTION_MODEL"], 
                          max_chars=6000, rag_k=8, collection_name=None):
    """Extract fields using LLM with RAG."""
    # Retrieve relevant chunks if collection is provided
    context_block = ""
    if collection_name:
        q = text[:2000]
        chunks = retrieve_similar_chunks(q, collection_name, top_k=rag_k)
        snippets = [doc for doc, meta, d in chunks if doc]
        if snippets:
            context_block = "\n\n---CONTEXT CHUNKS---\n\n" + "\n\n".join(snippets)
    
    # Build prompt
    body_text = text[:max_chars]
    prompt = get_extraction_prompt(category, body_text + ("\n\n" + context_block if context_block else ""))
    
    # Call LLM
    try:
        with st.spinner("Extracting fields with LLM..."):
            resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
            content = resp["message"]["content"].strip()
    except Exception as e:
        st.error(f"Extraction LLM failed: {e}")
        return {"AI Summary": "", "Note": f"LLM call failed: {e}"}
    
    # Clean and parse JSON
    content = re.sub(r"```(?:json)?", "", content).strip()
    
    try:
        return json.loads(content)
    except:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            jtxt = m.group(0)
            jtxt = re.sub(r",\s*}", "}", jtxt)
            jtxt = re.sub(r",\s*]", "]", jtxt)
            try:
                return json.loads(jtxt)
            except Exception:
                pass
    
    return {"AI Summary": content[:2000], "Note": "Failed to parse JSON"}

# Confidence scoring - COMPLETE FUNCTION
def compute_confidence_from_dict(extracted: dict, category: str = None) -> float:
    """Compute confidence score for extracted fields."""
    if not isinstance(extracted, dict) or len(extracted) == 0:
        return 0.0
    
    CATEGORY_HIGH = {
        "Claim Document": {
            "Primary Insured Name", "Policy Number", "Patient Name",
            "Hospital Name", "Admission Date", "Discharge Date",
            "Claim Amount", "Claim Type"
        },
        "Health Insurance Policy": {
            "Policy Holder Name", "Policy Number", "Insurance Company",
            "Sum Insured", "Policy Start Date", "Policy End Date"
        },
        "Motor Insurance Policy": {
            "Policy Holder Name", "Policy Number", "Registration Number",
            "Vehicle Model", "Policy Start Date", "Policy End Date"
        },
        "Life Insurance Policy": {
            "Policy Holder Name", "Policy Number", "Sum Assured",
            "Premium Amount", "Policy Term"
        },
        "IRCTC Ticket": {
            "Passenger Name", "Train Number", "PNR",
            "Date of Journey", "Boarding Station", "Destination Station"
        },
        "Invoice": {
            "Invoice Number", "Vendor Name", "Customer Name",
            "Invoice Date", "Total Amount"
        },
        "KYC / Identity Document": {
            "Name", "DOB", "Document Number"
        },
        "Hospital Bill": {
            "Hospital Name", "Total Amount", "Patient Name"
        },
        "Payment Receipt": {
            "Receipt Number", "Amount Paid", "Date"
        },
        "Legal Notice": {
            "Document Title", "Client / Complainant", "Accused / Respondent",
            "Issue Description", "Amount Pending", "Notice Date"
        }
    }
    
    CATEGORY_MEDIUM = {
        "Claim Document": {"TPA / Company ID", "Injury / Illness Type", "Billing Breakdown"},
        "Health Insurance Policy": {"Coverage Type", "TPA Name"},
        "Motor Insurance Policy": {"Engine / Chassis Number", "Coverage Type", "IDV"},
        "Life Insurance Policy": {"Nominee Name", "Nominee Relationship", "Policy Duration"},
        "IRCTC Ticket": {"Booking Status", "Current Status", "Coach / Seat"},
        "Invoice": {"Tax Amount", "Payment Terms", "Items"},
        "KYC / Identity Document": {"Address", "Gender", "Father's Name"},
        "Hospital Bill": {"Room Rent", "Consultation Fee", "Doctor Fee"},
        "Payment Receipt": {"Payment Mode", "Reference Number"},
        "Legal Notice": {"Advocate", "Contract Amount", "Payment Deadline", "Interest Rate"}
    }
    
    SKIP = {"AI Summary", "Note", "Document Type", "Key Fields"}
    
    # Get relevant keys for category
    high_keys = CATEGORY_HIGH.get(category, set())
    med_keys = CATEGORY_MEDIUM.get(category, set())
    
    total_weight = 0
    score = 0
    
    def is_valid(v):
        if v is None:
            return False
        s = str(v).strip()
        if not s or s.lower() == "not mentioned":
            return False
        if s in {"--", "-", "n/a", "xxx", "nil", "na", "null"}:
            return False
        if len(s) == 1 and not s.isalnum():
            return False
        return True
    
    # Scoring loop
    for field, value in extracted.items():
        if field in SKIP:
            continue
        
        # Handle nested dictionaries
        if isinstance(value, dict):
            for k2, v2 in value.items():
                w = 2 if k2 in med_keys else 1
                total_weight += w
                if is_valid(v2):
                    score += w
            continue
        
        # Handle lists
        if isinstance(value, list):
            total_weight += 1
            if value and len(value) > 0:
                # Check if list has valid items
                valid_items = [item for item in value if is_valid(item)]
                if len(valid_items) > 0:
                    score += 1
            continue
        
        # Determine weight
        if field in high_keys:
            weight = 3
        elif field in med_keys:
            weight = 2
        else:
            weight = 1
        
        total_weight += weight
        
        if is_valid(value):
            score += weight
    
    if total_weight == 0:
        return 0.5
    
    conf = score / total_weight
    return round(min(max(conf, 0.0), 1.0), 3)

# Main processing pipeline
def process_pdf_hybrid(pdf_path: str, collection_name: str, top_k=5, embed_threshold=0.30):
    """Main processing pipeline."""
    result_container = st.container()
    
    with result_container:
        st.subheader("Processing Pipeline")
        
        # 1) Extract raw text
        with st.status("Extracting text from PDF...", expanded=True) as status:
            pages = extract_text_auto(pdf_path)
            full_text = join_pages(pages)
            st.write(f"Extracted {len(full_text)} characters from {len(pages)} pages")
            status.update(label="Text extraction completed", state="complete")
        
        # 2) Ensure PDF is in Chroma
        with st.status("Vector database processing...", expanded=True) as status:
            client = get_chroma_client()
            try:
                col = client.get_collection(collection_name)
                st.write(f"Using existing collection: {collection_name}")
            except Exception:
                col = upsert_pdf_to_chroma(pdf_path, collection_name)
                if col:
                    st.write(f"Created new collection with {col.count()} chunks")
            status.update(label="Vector database ready", state="complete")
        
        # 3) Retrieve chunks for classification
        with st.status("Retrieving relevant chunks...", expanded=True) as status:
            q = full_text[:2000]
            top_chunks = retrieve_similar_chunks(q, collection_name, top_k=top_k)
            st.write(f"Retrieved {len(top_chunks)} relevant chunks")
            status.update(label="Chunks retrieved", state="complete")
        
        # 4) Category detection
        with st.status("Classifying document...", expanded=True) as status:
            candidates = embedding_candidate_category(top_chunks)
            
            if candidates and candidates[0][1] >= embed_threshold:
                category = candidates[0][0]
                classifier_conf = float(candidates[0][1])
                st.write(f"Embedding-based classification: **{category}** (score: {classifier_conf:.2f})")
            else:
                snippets = [doc for doc, meta, dist in top_chunks if doc]
                cls = llm_classify_with_context(
                    full_text,
                    model=config["CLASSIFIER_MODEL"],
                    top_k_snippets=snippets
                )
                category = cls.get("category", "Other")
                classifier_conf = float(cls.get("confidence", 0.0))
                st.write(f"LLM-based classification: **{category}** (confidence: {classifier_conf:.2f})")
            status.update(label="Classification completed", state="complete")
        
        # 5) Field extraction
        with st.status("Extracting structured fields...", expanded=True) as status:
            extracted = extract_fields_via_llm(
                full_text,
                category,
                model=config["EXTRACTION_MODEL"],
                max_chars=6000,
                collection_name=collection_name
            )
            st.write(f"Extracted {len(extracted)} fields")
            status.update(label="Field extraction completed", state="complete")
        
        # 6) Confidence scoring
        with st.status("Computing confidence scores...", expanded=True) as status:
            fields_conf = compute_confidence_from_dict(extracted, category)
            combined_conf = round((classifier_conf + fields_conf) / 2, 3)
            st.write(f"Fields confidence: {fields_conf:.3f}")
            st.write(f"Combined confidence: {combined_conf:.3f}")
            status.update(label="Confidence scoring completed", state="complete")
        
        # 7) Prepare result
        result = {
            "document_path": pdf_path,
            "collection_name": collection_name,
            "category": category,
            "classification_confidence": float(classifier_conf),
            "fields_confidence": fields_conf,
            "combined_confidence": combined_conf,
            "extracted": extracted
        }
        
        return result

# Streamlit UI
def main():
    st.title("📄 Document Intelligence System")
    st.markdown("""
    This system extracts structured information from various documents using:
    - **OCR** for text extraction
    - **Vector embeddings** for semantic search
    - **LLM** for classification and extraction
    - **RAG** for contextual information retrieval
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Collection management
        st.subheader("Vector Database")
        
        # Display current collection
        if st.session_state.collection_name:
            st.info(f"**Current Collection:**\n`{st.session_state.collection_name}`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Current Collection", type="secondary"):
                if st.session_state.collection_name:
                    client = get_chroma_client()
                    if client:
                        try:
                            col = client.get_collection(st.session_state.collection_name)
                            data = col.get()
                            ids = data.get("ids", [])
                            if ids:
                                col.delete(ids=ids)
                                st.success(f"Reset {len(ids)} items")
                            else:
                                st.info("Collection already empty")
                        except Exception as e:
                            st.error(f"Reset failed: {e}")
        
        with col2:
            if st.button("List All Collections", type="secondary"):
                client = get_chroma_client()
                if client:
                    collections = client.list_collections()
                    if collections:
                        st.write("Available collections:")
                        for c in collections:
                            st.write(f"- **{c.name}**")
                            if c.name in st.session_state.all_collections:
                                info = st.session_state.all_collections[c.name]
                                st.write(f"  File: {info['filename']}, Chunks: {info['chunks']}")
                    else:
                        st.info("No collections found")
        
        # Processing parameters
        st.subheader("Processing Parameters")
        top_k = st.slider("Top K chunks for RAG", 1, 20, 5)
        embed_threshold = st.slider("Embedding threshold", 0.0, 1.0, 0.30, 0.05)
        
        st.divider()
        
        # System info
        st.subheader("System Information")
        st.write(f"Embedding model: {config['EMBED_MODEL_NAME']}")
        st.write(f"Classifier model: {config['CLASSIFIER_MODEL']}")
        st.write(f"Extraction model: {config['EXTRACTION_MODEL']}")
        
        # Ollama status
        try:
            models = ollama.list()
            st.success("✅ Ollama connected")
            st.write(f"Available models: {len(models['models'])}")
        except:
            st.error("❌ Ollama not available")
    
    # Main area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "🔍 Search", "📈 Analytics"])
    
    with tab1:
        st.subheader("Upload Document")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Get file hash and check if already processed
            file_content = uploaded_file.getvalue()
            current_file_hash = get_file_hash(file_content)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Generate unique collection name
            collection_name = generate_collection_name(uploaded_file.name, file_content)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**File Information:**")
                st.write(f"Name: {uploaded_file.name}")
                st.write(f"Size: {uploaded_file.size / 1024:.1f} KB")
                st.write(f"Collection: `{collection_name}`")
            
            with col2:
                process_button = st.button("✨ Process Document", type="primary", use_container_width=True)
                
                # Check if this is a new file
                if st.session_state.file_hash != current_file_hash:
                    st.info("📄 New file detected. Ready to process.")
                
                if process_button:
                    with st.spinner("Processing document..."):
                        # Update session state
                        st.session_state.uploaded_file = tmp_path
                        st.session_state.file_hash = current_file_hash
                        st.session_state.collection_name = collection_name
                        
                        result = process_pdf_hybrid(
                            tmp_path,
                            collection_name=collection_name,
                            top_k=top_k,
                            embed_threshold=embed_threshold
                        )
                        st.session_state.extraction_result = result
                        st.success("Processing complete!")
                        st.rerun()
    
    with tab2:
        st.subheader("Extraction Results")
        
        if st.session_state.extraction_result:
            result = st.session_state.extraction_result
            
            # Show collection info
            if "collection_name" in result:
                st.info(f"**Document Collection:** `{result['collection_name']}`")
            
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Document Category", result["category"])
            with col2:
                st.metric("Classification Confidence", f"{result['classification_confidence']:.2%}")
            with col3:
                st.metric("Fields Confidence", f"{result['fields_confidence']:.2%}")
            with col4:
                st.metric("Overall Confidence", f"{result['combined_confidence']:.2%}")
            
            # Confidence visualization
            fig = go.Figure(data=[
                go.Bar(name='Confidence Scores',
                       x=['Classification', 'Fields', 'Overall'],
                       y=[result['classification_confidence'], 
                          result['fields_confidence'], 
                          result['combined_confidence']],
                       text=[f"{result['classification_confidence']:.1%}", 
                             f"{result['fields_confidence']:.1%}", 
                             f"{result['combined_confidence']:.1%}"],
                       textposition='auto')
            ])
            fig.update_layout(yaxis_range=[0, 1], title="Confidence Scores")
            st.plotly_chart(fig, use_container_width=True)
            
            # Extracted data
            st.subheader("Extracted Fields")
            
            extracted = result["extracted"]
            
            # Create tabs for different field groups
            field_tabs = st.tabs(["📋 All Fields", "📄 AI Summary", "💾 Export"])
            
            with field_tabs[0]:
                # Display extracted fields in a structured way
                for key, value in extracted.items():
                    if isinstance(value, dict):
                        with st.expander(f"📂 {key}"):
                            for subkey, subvalue in value.items():
                                st.write(f"**{subkey}:** {subvalue}")
                    elif isinstance(value, list):
                        with st.expander(f"📋 {key} ({len(value)} items)"):
                            for i, item in enumerate(value):
                                st.write(f"{i+1}. {item}")
                    else:
                        if key == "AI Summary":
                            st.info(f"**{key}:** {value}")
                        else:
                            st.write(f"**{key}:** {value}")
            
            with field_tabs[1]:
                if "AI Summary" in extracted:
                    st.info(extracted["AI Summary"])
                else:
                    st.warning("No AI Summary found in extracted data")
            
            with field_tabs[2]:
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"extraction_{result.get('collection_name', 'result')}.json",
                    mime="application/json"
                )
                
                # Display JSON
                with st.expander("View Raw JSON"):
                    st.json(result)
        else:
            st.info("Upload and process a document to see results here.")
    
    with tab3:
        st.subheader("Semantic Search")
        
        if st.session_state.collection_name:
            st.info(f"Searching in collection: `{st.session_state.collection_name}`")
            
            query = st.text_input("Search query", placeholder="Enter text to search in your document...")
            
            if query:
                with st.spinner("Searching..."):
                    chunks = retrieve_similar_chunks(query, st.session_state.collection_name, top_k=3)
                    
                    if chunks:
                        st.write(f"Found {len(chunks)} relevant chunks:")
                        
                        for i, (doc, meta, dist) in enumerate(chunks):
                            with st.expander(f"Chunk {i+1} (distance: {dist if dist else 'N/A'})"):
                                st.write(f"**Source:** {meta.get('source', 'Unknown')}")
                                st.write(f"**Collection:** {meta.get('collection', 'N/A')}")
                                st.write(f"**Chunk index:** {meta.get('chunk_index', 'N/A')}")
                                st.divider()
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                    else:
                        st.warning("No relevant chunks found.")
        else:
            st.info("Please upload and process a document first to enable search.")
    
    with tab4:
        st.subheader("Analytics & Insights")
        
        if st.session_state.extraction_result:
            result = st.session_state.extraction_result
            extracted = result["extracted"]
            
            # Show collection stats
            st.write("### Collection Information")
            if "collection_name" in result:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Collection Name", result["collection_name"])
                with col2:
                    if result["collection_name"] in st.session_state.all_collections:
                        info = st.session_state.all_collections[result["collection_name"]]
                        st.metric("Total Chunks", info["chunks"])
            
            # Field completion analysis
            st.write("### Field Completion Analysis")
            
            # Count filled vs empty fields
            def count_fields(data, prefix=""):
                filled = 0
                total = 0
                
                for key, value in data.items():
                    if key in ["AI Summary", "Note"]:
                        continue
                    
                    if isinstance(value, dict):
                        sub_filled, sub_total = count_fields(value, f"{prefix}{key}.")
                        filled += sub_filled
                        total += sub_total
                    elif isinstance(value, list):
                        total += 1
                        if value and len(value) > 0:
                            filled += 1
                    else:
                        total += 1
                        if value and str(value).strip() and str(value).lower() != "not mentioned":
                            filled += 1
                
                return filled, total
            
            filled_count, total_count = count_fields(extracted)
            completion_rate = filled_count / total_count if total_count > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filled Fields", filled_count)
            with col2:
                st.metric("Completion Rate", f"{completion_rate:.1%}")
            
            # Create completion chart
            fig = go.Figure(data=[
                go.Pie(labels=['Filled', 'Empty'],
                       values=[filled_count, total_count - filled_count],
                       hole=.3)
            ])
            fig.update_layout(title="Field Completion")
            st.plotly_chart(fig, use_container_width=True)
            
            # Document type distribution (if multiple documents processed)
            st.write("### Document Insights")
            
            # Extract key-value pairs for display
            key_data = []
            for key, value in extracted.items():
                if key not in ["AI Summary", "Note"] and not isinstance(value, (dict, list)):
                    key_data.append({
                        "Field": key,
                        "Value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value),
                        "Length": len(str(value))
                    })
            
            if key_data:
                df = pd.DataFrame(key_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Process a document to see analytics here.")

if __name__ == "__main__":
    main()