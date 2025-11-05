# app/ingest_qdrant.py
import os, time, json, argparse, unicodedata, uuid, re
from pathlib import Path
from statistics import fmean
import httpx
from pypdf import PdfReader

# -------------------------------------------------------------------
# Normalización y limpieza
# -------------------------------------------------------------------
def normalize_text(txt: str) -> str:
    # Une palabras cortadas al final de línea: "visi-\nón" -> "visión"
    txt = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", txt, flags=re.UNICODE)

    # Convierte saltos duros en espacios cuando no terminan oración
    txt = re.sub(r"(?<![.!?:])\s*\n\s*(?!\n)", " ", txt)

    # Colapsa espacios múltiples
    txt = re.sub(r"[ \t]{2,}", " ", txt)

    # Normaliza saltos de párrafo a doble \n
    txt = re.sub(r"\n{3,}", "\n\n", txt.strip())

    return txt

def clean_page_artifacts(txt: str) -> str:
    """
    Elimina artefactos simples típicos de PDFs:
    - líneas que son solo números (número de página)
    - patrones 'Página 12', 'Pág. 7'
    - encabezados/pies repetidos cortos (heurística ligera)
    """
    lines = [l for l in txt.splitlines()]

    # Quita líneas que son solo números (1-3 dígitos)
    lines = [l for l in lines if not re.fullmatch(r"\s*\d{1,3}\s*", l)]

    # Quita 'Página X' o 'Pág. X'
    lines = [re.sub(r"\bP(á|a)g(?:ina)?\.?\s*\d{1,4}\b", "", l, flags=re.IGNORECASE) for l in lines]

    # Heurística: elimina líneas repetidas cortas (posibles encabezados)
    freq = {}
    for l in lines:
        s = l.strip()
        if 3 <= len(s) <= 60:
            freq[s] = freq.get(s, 0) + 1
    repeated = {s for s, c in freq.items() if c >= 4}  # aparece muchas veces
    lines = [l for l in lines if l.strip() not in repeated]

    out = "\n".join(lines)
    # Quita espacios extra finales tras limpieza
    out = re.sub(r"[ \t]+(\n|$)", r"\1", out)
    return out

# -------------------------------------------------------------------
# Embeddings locales (fallback o directo)
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False
    SentenceTransformer = None

# HTML opcional
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

from app.settings import get_settings
S = get_settings()

# ---------- HF config ----------
HF_API_TOKEN = (S.HF_API_TOKEN or "").strip()
#HF_MODEL_NAME = (S.HF_EMBED_MODEL or "intfloat/multilingual-e5-small").strip()
HF_MODEL_NAME = "intfloat/multilingual-e5-small"

# 1) Router nuevo  2) Endpoint clásico (fallback)
HF_URLS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_NAME}",
    f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
]
HF_HEADERS_BASE = {
    "Content-Type": "application/json",
    "X-Task": "feature-extraction",
    "X-Wait-For-Model": "true",
}
HF_HEADERS_AUTH = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# ---------- Qdrant config ----------
QDRANT_URL = (S.QDRANT_URL or "http://localhost:6333").rstrip("/")
QDRANT_COLLECTION = S.QDRANT_COLLECTION
QDRANT_API_KEY = (S.QDRANT_API_KEY or "").strip()
_Q_HEADERS = {"Content-Type": "application/json"}
if QDRANT_API_KEY:
    _Q_HEADERS["api-key"] = QDRANT_API_KEY

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".html", ".htm"}

# ---------- util archivos ----------
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def safe_extract_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path, strict=False)
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out)
    except Exception as e:
        print(f"[WARN] No se pudo leer '{pdf_path}': {e}")
        return ""

def load_text_from_file(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return safe_extract_text(path)
    if lower.endswith((".txt", ".md")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] No se pudo leer texto '{path}': {e}")
            return ""
    if lower.endswith((".html", ".htm")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            if _HAS_BS4:
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)
            return html
        except Exception as e:
            print(f"[WARN] No se pudo leer HTML '{path}': {e}")
            return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] No se pudo leer archivo '{path}': {e}")
            return ""

def iter_source_files(base_dir: str):
    base = Path(_nfc(base_dir)).resolve()
    if not base.exists():
        print(f"[WARN] Carpeta no existe: {base}")
        return
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p

# -------------------------------------------------------------------
# Chunking consciente de oraciones
# -------------------------------------------------------------------
_SENT_END = re.compile(r"([.!?…]+)(\s+|$)")

def _split_sentences(text: str) -> list[str]:
    """
    División simple por signos de fin de oración.
    Conserva los signos y evita romper palabras.
    """
    text = text.strip()
    if not text:
        return []
    parts, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        sent = text[start:end].strip()
        if sent:
            parts.append(sent)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120, min_chunk_chars: int = 120) -> list[str]:
    """
    Construye chunks acumulando oraciones hasta ~max_chars.
    Si una oración es demasiado larga, cae a corte por palabras.
    """
    text = (text or "").strip()
    if not text:
        return []

    sents = _split_sentences(text)
    if not sents:
        sents = [text]

    chunks, cur = [], ""
    for s in sents:
        if len(s) > max_chars:
            # Cae a corte por palabras dentro de la oración
            words = s.split()
            tmp = ""
            for w in words:
                if len(tmp) + 1 + len(w) > max_chars:
                    if tmp:
                        chunks.append(tmp.strip())
                    # solapamiento básico entre cortes largos
                    tmp = w
                else:
                    tmp = (tmp + " " + w).strip()
            if tmp:
                chunks.append(tmp.strip())
            cur = ""  # reinicia acumulador
            continue

        if len(cur) + 1 + len(s) <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur.strip())
            # solapamiento: añade cola del chunk anterior
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s

    if cur:
        chunks.append(cur.strip())

    # filtra muy pequeños (ruido)
    chunks = [c for c in chunks if len(c) >= min_chunk_chars or len(chunks) == 1]
    return chunks

# ---------- HF embeddings ----------
def _mean_pooling(nested):
    if not nested:
        return []
    if isinstance(nested[0], (list, tuple)):
        dim = len(nested[0])
        return [fmean(row[i] for row in nested) for i in range(dim)]
    return list(nested)

def _e5_prefix(text: str, kind: str) -> str:
    if "e5" in HF_MODEL_NAME.lower() or "intfloat/" in HF_MODEL_NAME.lower():
        return f"{'query' if kind=='query' else 'passage'}: {text}"
    return text

# === Inicializa el modelo local ANTES de hf_embed ===
_LOCAL_EMBED_MODEL = None
if _SBERT_AVAILABLE:
    try:
        _LOCAL_EMBED_MODEL = SentenceTransformer(HF_MODEL_NAME)
        print(f"[EMB] Modelo local cargado: {HF_MODEL_NAME}")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo local '{HF_MODEL_NAME}': {e}")
        _LOCAL_EMBED_MODEL = None

def hf_embed(text: str, kind: str = "passage", retries: int = 3, timeout=45.0) -> list[float]:
    """
    1) LOCAL con sentence-transformers (rápido).
    2) Remoto HF (router -> clásico) si hay token.
    """
    # --- 1) LOCAL ---
    if _LOCAL_EMBED_MODEL is not None:
        txt = _e5_prefix(text, kind)
        try:
            vec = _LOCAL_EMBED_MODEL.encode([txt], normalize_embeddings=True)
            return vec[0].tolist()
        except Exception as e:
            print(f"[WARN] Falla embedding local, intento remoto: {e}")

    # --- 2) REMOTO (HF) ---
    if not HF_API_TOKEN:
        raise RuntimeError("No hay modelo local disponible y falta HF_API_TOKEN para remoto.")
    payload = {"inputs": _e5_prefix(text, kind)}
    last_err = None
    for attempt in range(1, retries + 1):
        for url in HF_URLS:
            try:
                with httpx.Client(timeout=timeout) as client:
                    r = client.post(
                        url,
                        headers={**HF_HEADERS_BASE, **HF_HEADERS_AUTH},
                        json=payload,
                    )
                    if r.status_code in (503, 429):
                        time.sleep(2 * attempt); continue
                    if r.status_code >= 400:
                        print(f"[HF] {url} → {r.status_code}: {r.text[:300]}")
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and "error" in data:
                        raise RuntimeError(f"HuggingFace error: {data['error']}")
                    vec = _mean_pooling(data)
                    if not vec:
                        raise RuntimeError("Embedding vacío recibido desde HF.")
                    return vec
            except Exception as e:
                last_err = e
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"No se pudo obtener embedding tras {retries} intentos. Último error: {last_err}")

def qdrant_dim() -> int:
    # dimensión típica de e5-small = 384 (lo calculamos en caliente)
    return len(hf_embed("ping", kind="query"))

# ---------- Qdrant REST ----------
QDRANT_VECTOR_DISTANCE = "Cosine"  # puedes cambiar a "Dot" o "Euclid" si prefieres

def ensure_collection(recreate: bool, dim: int):
    with httpx.Client(timeout=30) as client:
        if recreate:
            client.delete(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", headers=_Q_HEADERS)
            print(f"[QDRANT] Colección eliminada (si existía): {QDRANT_COLLECTION}")
        body = {
            "vectors": {"size": dim, "distance": QDRANT_VECTOR_DISTANCE},
            "on_disk_payload": True
        }
        r = client.put(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", headers=_Q_HEADERS, json=body)
        r.raise_for_status()
        print(f"[QDRANT] Colección lista: {QDRANT_COLLECTION} (dim={dim}, dist={QDRANT_VECTOR_DISTANCE})")

def upsert_batch(points):
    if not points:
        return
    body = {"points": points}
    with httpx.Client(timeout=60) as client:
        r = client.put(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points", headers=_Q_HEADERS, json=body)
        try:
            r.raise_for_status()
        except Exception:
            print("[ERROR] Upsert falló:", r.text[:400])
            raise
        else:
            print("[OK] Upsert:", r.json().get("status", "ok"))

def count_points() -> int:
    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/count",
            headers=_Q_HEADERS, json={"exact": True}
        )
        r.raise_for_status()
        data = r.json()
        print("[QDRANT] Conteo final:", data)
        return int(data.get("count", 0))

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Ingesta CCP → Qdrant (E5 embeddings)")
    parser.add_argument("--dir", default="knowledge/ccp", help="Carpeta con documentos")
    parser.add_argument("--recreate", action="store_true", help="Borra y recrea la colección")
    parser.add_argument("--batch", type=int, default=64, help="Tamaño de lote")
    parser.add_argument("--max-chars", type=int, default=800, help="Tamaño máximo por chunk")
    parser.add_argument("--overlap", type=int, default=120, help="Solapamiento entre chunks")
    parser.add_argument("--min-chars", type=int, default=120, help="Mínimo por chunk (ruido debajo de esto)")
    args = parser.parse_args()

    print(f"[ENV] QDRANT_URL={QDRANT_URL!r}  COLLECTION={QDRANT_COLLECTION!r}")
    print(f"[ENV] HF_MODEL_NAME={HF_MODEL_NAME!r}")

    dim = qdrant_dim()
    ensure_collection(args.recreate, dim)

    points, total_chunks = [], 0
    for path in iter_source_files(args.dir):
        file = str(path)
        raw = load_text_from_file(file)
        if not raw.strip():
            print(f"[WARN] Sin texto útil, se omite: {path.name}")
            continue

        # 🔹 Limpieza y normalización SIEMPRE antes de chunking
        cleaned = clean_page_artifacts(raw)
        cleaned = normalize_text(cleaned)

        subs = chunk_text(
            cleaned,
            max_chars=args.max_chars,
            overlap=args.overlap,
            min_chunk_chars=args.min_chars
        )
        if not subs:
            print(f"[WARN] Sin chunks después de limpiar: {path.name}")
            continue

        print(f"[INFO] {path.name} → {len(subs)} chunks")
        for idx, sub in enumerate(subs):
            emb = hf_embed(sub, kind="passage")
            points.append({
                "id": str(uuid.uuid4()),
                "vector": emb,
                "payload": {
                    "path": file,
                    "filename": path.name,
                    "chunk_index": idx,
                    "text": sub
                }
            })
            total_chunks += 1
            if len(points) >= args.batch:
                upsert_batch(points)
                points.clear()

    if points:
        upsert_batch(points)
    cnt = count_points()
    print(f"[DONE] Ingesta finalizada ✅ (chunks subidos: {total_chunks}, en colección: {cnt})")

if __name__ == "__main__":
    main()
