# app/rag.py
import re, httpx, unicodedata
from typing import List, Dict, Any, Optional

from app.settings import get_settings
from app.ingest_qdrant import hf_embed

S = get_settings()
QDRANT_URL = S.QDRANT_URL.rstrip("/")
COLL = S.QDRANT_COLLECTION
GROQ_KEY = (S.GROQ_API_KEY or "").strip()
GROQ_MODEL = S.GROQ_MODEL or "gemma2-9b-it"

# ---------------- Reranker opcional ----------------
try:
    from sentence_transformers import CrossEncoder
    _HAS_XENC = True
    _XENC = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=384)
except Exception:
    _HAS_XENC = False
    _XENC = None

# ---------------- SYSTEM (conciso anti-alucinación) ----------------
SYSTEM = """
    Eres **Asistente CCP**, el chatbot oficial de la **Cámara de Comercio de Pamplona (CCP)**, Colombia.
    Tu función es brindar información clara, precisa y actualizada sobre los servicios, trámites y actividades de la CCP.
    Responde **exclusivamente** sobre la Cámara de Comercio de Pamplona (Norte de Santander, Colombia). No hables de otras cámaras ni de otras ciudades.

    ## Estilo conversacional
    - Tono: cordial, profesional y natural, como si conversaras por WhatsApp.
    - Usa frases cortas, con formato simple y ordenado pero con informacion completa.
    - Si el usuario comete errores ortográficos, entiende la intención y responde correctamente.
    - Si el usuario pregunta varias cosas a la vez (por ejemplo, “dirección y horario”), responde en secciones separadas con encabezados en **negrita**.
    - Cierra con una frase útil y fluida (por ejemplo: “¿Deseas que te indique *Trámites* o *Certificados*?”).

    ## Principios
    1. **Dominio:** Solo información de la CCP Pamplona. Si el texto fuente o la pregunta menciona otra ciudad o entidad (Bogotá, Bucaramanga, Cali, Medellín, Cartagena, etc.), ignóralo.
    2. **Exactitud:** Usa exclusivamente el CONTEXTO provisto. No inventes datos.
    3. **Veracidad:** Si no tienes información en el contexto, di: “No cuento con ese dato en este momento.” y sugiere contactar a un asesor.
    4. **Precisión:** Responde únicamente lo que se te pregunta; evita extenderte innecesariamente.
    5. **Coherencia:** Nunca combines temas distintos si no fueron solicitados.
    6. **Privacidad:** No inventes correos, teléfonos ni direcciones; usa solo los institucionales que aparezcan en el contexto.
    7. **Brevedad:** Máximo tres oraciones por sección; si una sección es larga, resume.
    8. **No alucines:** No crees información, no cites fuentes inexistentes.

    ## Tipos de intención que debes detectar
    - **ubicacion:** dirección o sede.
    - **horarios:** atención al público.
    - **telefonos:** teléfonos, WhatsApp, correos.
    - **mision_vision:** misión, visión, valores o política de calidad.
    - **tramites:** matrícula, renovación, cancelación, ESAL, requisitos, pasos.
    - **tarifas:** costos, valores o aranceles.
    - **conciliacion:** información del Centro de Conciliación.
    - **eventos:** capacitaciones, afiliaciones, actividades.
    - **general:** información institucional o dudas no clasificadas.

    ## Formato de respuesta
    - Si hay **una sola intención**, responde solo a esa.
    - Si hay **varias intenciones**, usa secciones con títulos en **negrita**:
    **Dirección:** …  
    **Horarios:** …  
    **Teléfonos/WhatsApp:** …  
    **Correo:** …  
    - Si falta un dato, indica: “No cuento con ese dato en este momento.”
    - Nunca uses más de tres oraciones por sección.
    - Mantén la estructura ordenada y agradable a la vista.

    ## Ejemplos correctos
    Usuario: “¿Dónde están ubicados?”  
    ✅ Respuesta: “**Dirección:** Carrera 5 No. 5-88, Pamplona, Norte de Santander, Colombia.”

    Usuario: “necesito dirección y horarios”  
    ✅ Respuesta:  
    **Dirección:** Carrera 5 No. 5-88, Pamplona, Norte de Santander.  
    **Horarios:** Lunes a viernes, 8:00 a. m.–12:00 m. y 2:00 p. m.–6:00 p. m.  
    ¿Deseas que te indique *Trámites* o *Certificados*?

    Usuario: “misión y visión”  
    ✅ Respuesta:  
    **Misión:** Promover el desarrollo empresarial y económico en Pamplona mediante servicios de registro, formación y apoyo al emprendimiento.  
    **Visión:** Ser una entidad líder en la prestación de servicios empresariales en Norte de Santander.

    ## Ejemplos incorrectos
    ❌ Responder con información de otras cámaras o ciudades.  
    ❌ Escribir párrafos largos o con datos irrelevantes.  
    ❌ Suponer horarios o tarifas sin respaldo del contexto.

    Tu meta es mantener una conversación **fluida, precisa y 100 % enfocada en la Cámara de Comercio de Pamplona.**
    """


# ===================== Utilidades de texto / fuzzy =====================
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def _norm_text(s: str) -> str:
    s = _strip_accents((s or "").lower().strip())
    s = re.sub(r"\s+", " ", s)
    return s

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))
        prev = cur
    return prev[-1]

def _fuzzy_hit(text: str, keyword: str, max_dist: int = 1) -> bool:
    t = _norm_text(text)
    k = _norm_text(keyword)
    if k in t:
        return True
    tokens = t.split()
    ktoks = k.split()
    if len(ktoks) == 1:
        kk = ktoks[0]
        for w in tokens:
            if _levenshtein(w, kk) <= (2 if len(kk) >= 6 else max_dist):
                return True
        return False
    for i in range(len(tokens) - len(ktoks) + 1):
        win = tokens[i:i+len(ktoks)]
        d = sum(_levenshtein(w, kk) for w, kk in zip(win, ktoks))
        tol = sum(2 if len(kk) >= 6 else 1 for kk in ktoks)
        if d <= tol:
            return True
    return False

def _any_fuzzy(text: str, keywords: List[str]) -> bool:
    return any(_fuzzy_hit(text, kw) for kw in keywords)

# ---------------- Small talk (con fuzzy) ----------------
_SMALLTALK_HELLOS = ["hola", "buenos dias", "buenas", "buenas tardes", "buenas noches", "que tal", "qué tal", "hey"]
_SMALLTALK_THANKS = ["gracias", "muchas gracias", "ok", "listo", "perfecto", "dale"]
_SMALLTALK_BYES = ["adios", "adiós", "chao", "hasta luego", "nos vemos", "bye"]

def _smalltalk_reply(q: str) -> Optional[str]:
    t = q or ""
    if _any_fuzzy(t, _SMALLTALK_HELLOS):
        return "¡Hola! 👋 Soy el asistente de la Cámara de Comercio de Pamplona. ¿Qué necesitas: *Ubicación*, *Horarios*, *Teléfonos* o *Trámites*?"
    if _any_fuzzy(t, _SMALLTALK_THANKS):
        return "¡Con gusto! 😊 ¿Te indico *Trámites* o *Certificados*?"
    if _any_fuzzy(t, _SMALLTALK_BYES):
        return "¡Hasta luego! 🙌"
    return None

# ---------------- Similaridad y selección ----------------
def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den1 = sum(x * x for x in a) ** 0.5
    den2 = sum(y * y for y in b) ** 0.5
    return (num / (den1 * den2)) if den1 and den2 else 0.0

def _mmr_select(cands, lam=0.7, k=6):
    sel, pool = [], cands[:]
    while pool and len(sel) < k:
        best, best_score = None, -1e9
        for it in pool:
            rel = float(it.get("score", 0.0))
            div = 0.0
            if sel and it.get("vector"):
                div = max(_cos(it["vector"], s.get("vector", [])) for s in sel if s.get("vector"))
            mmr = lam * rel - (1 - lam) * div
            if mmr > best_score:
                best, best_score = it, mmr
        sel.append(best); pool.remove(best)
    return sel

# ---------------- Intenciones ----------------
def _detect_intents_mixed(q: str) -> List[str]:
    groups = [
        ("ubicacion",    ["ubicacion", "ubic", "direccion", "dirección", "sede", "mapa", "donde", "dónde", "estan", "están"]),
        ("horarios",     ["horario", "abre", "cierra", "jornada", "sabado", "sábado", "domingo", "festivo", "hora"]),
        ("telefonos",    ["telefono", "teléfono", "pbx", "whatsapp", "numero", "número", "linea", "línea", "correo", "email", "contacto"]),
        ("mision_vision",["mision", "misión", "vision", "visión", "valores", "politica de calidad", "política de calidad"]),
        ("tramites",     ["tramite", "trámite", "matricula", "matrícula", "renovacion", "renovación", "cancelacion", "cancelación", "esal", "registro"]),
        ("tarifas",      ["tarifa", "tarifario", "precio", "costo", "costos", "valor", "valores"]),
        ("conciliacion", ["conciliacion", "conciliación", "audiencia", "apoderado"]),
        ("eventos",      ["evento", "capacitacion", "capacitaciones", "afiliacion", "afiliación"]),
    ]
    found = []
    for name, kws in groups:
        if _any_fuzzy(q, kws):
            found.append(name)
    order = ["ubicacion","horarios","telefonos","mision_vision","tramites","tarifas","conciliacion","eventos"]
    return [x for x in order if x in found] or ["general"]

# ---------------- Extracción determinista ----------------
RE_ADDRESS = re.compile(r"(?:^|\n)\s*Direcci[oó]n[:\s]*([^\n]+)", re.I)
PHONE_MOBILE = re.compile(r"(?:\+57\s*)?3\d{2}[\s\-.]?\d{3}[\s\-.]?\d{4}\b")
PHONE_LAND   = re.compile(r"(?:\+57\s*)?(?:\(60[1-8]\)|60[1-8])[\s\-.]?\d{3}[\s\-.]?\d{4}\b")
RE_EMAIL     = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)

# Horarios: cualquier línea que contenga horas, pero se limpiará luego
RE_HOURS_RAW = re.compile(r"(?i)(horario[s]?:?.{0,40})?(\d{1,2}[:\.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?))[^.\n]*")

# Ruido administrativo típico (se elimina)
RE_NOISE = re.compile(
    r"(?i)\b(N[uú]mero de|jurisdicci[oó]n|cobertura|indicador|tiempo de respuesta|vigencia|art[íi]culo|municipios|comparativo|cooperativa)\b.*"
)

_CURRENCY  = re.compile(r"(\$\s?\d[\d\.\,]*\b|\b\d[\d\.\,]*\s?(?:COP|col|pesos))", re.I)
_TARIFA_KW = re.compile(r"(tarifa(?:rio)?|valor(?:es)?|costo(?:s)?|precio(?:s)?|pago|arancel|certificad[oa]s?)", re.I)
_REQ_KW    = re.compile(r"(requisit|document|paso|procedimiento|c[oó]mo|como|plazo|d[oó]nde|donde|solicita[rc])", re.I)

CITY_BLOCKLIST = re.compile(r"\b(bogota|bogotá|bucaramanga|cali|medellin|medellín|cartagena|barranquilla|manizales|cucuta|cúcuta)\b", re.I)

def _truncate(s: str, n: int = 160) -> str:
    s = (s or "").strip()
    if len(s) <= n: return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + "…"

def _clean_text(t: str) -> str:
    if not t: return ""
    t = RE_NOISE.sub("", t)                # quita bloques de indicadores/estadísticas
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # une cortes de línea en PDFs
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t

def _clean_address(addr: str) -> str:
    addr = re.sub(r"https?://\S+", "", addr or "")
    addr = re.split(r"(Tel[eé]fono|PBX|WhatsApp|Correo|Horario|Historia|Edificio)\b", addr, maxsplit=1)[0]
    addr = addr.replace(" .", ".").strip().strip(",;:·-–—")
    addr = re.sub(r"\([^)]{40,}\)", "", addr).strip()
    return _truncate(addr, 120)

RE_ADDRESS_FALLBACK = re.compile(
    r"((?:Carrera|Calle|Av(?:enida)?|Kr\.?|Cl\.?|Transversal)\s*[\wº°#\.\-\s]+?\d[\wº°#\.\-\s,]*"
    r"(?:Pamplona|Norte de Santander)?[^\n]*)",
    re.I,
)

def _norm_phone(p: str) -> str:
    d = re.sub(r"[^\d]", "", p or "")
    if len(d) >= 10 and d[-10] == "3":  # móvil
        last10 = d[-10:]
        return f"+57 {last10[0:3]} {last10[3:6]} {last10[6:10]}"
    if len(d) >= 11 and d[-11:-8] == "607":
        last11 = d[-11:]
        return f"(607) {last11[3:6]} {last11[6:10]}"
    if len(d) >= 11 and d[-11:-9] == "60":
        last11 = d[-11:]
        return f"(60{last11[2]}) {last11[3:6]} {last11[6:10]}"
    return ""
def _canon_hours(raw_block: str) -> List[str]:
    """
    Extrae horarios claros y compactos del texto, evitando parrafadas largas.
    Devuelve máximo 2 líneas legibles.
    """
    txt = _clean_text(raw_block or "")
    if not txt:
        return []

    # Buscar líneas con hora o palabras clave relevantes
    pattern = re.compile(
        r"(?i)(lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|festivo|hora|horario)[^.\n]{0,80}(\d{1,2}[:\.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?))[^.\n]{0,60}"
    )
    matches = pattern.findall(txt)
    lines = []

    # Unir coincidencias contiguas si forman un bloque horario coherente
    joined = []
    for m in pattern.finditer(txt):
        seg = m.group(0)
        seg = re.sub(r"\s{2,}", " ", seg)
        seg = seg.replace(" .", ".").strip(" .,:;")
        if 20 < len(seg) < 180:   # rango más flexible
            joined.append(seg)

    # Si no hay coincidencias útiles, usar formato fijo por defecto
    if not joined:
        return ["Lunes a viernes: 8:00 a. m.–12:00 m. y 2:00 p. m.–6:00 p. m."]

    # Filtrar duplicados y limitar a máximo 2 líneas
    seen, out = set(), []
    for h in joined:
        k = h.lower()
        if k not in seen:
            out.append(h)
            seen.add(k)
        if len(out) >= 2:
            break
    return out


def _extract_contact_block(text: str) -> Dict[str, Any]:
    txt = _clean_text(text or "")

    # Dirección
    address = None
    m = RE_ADDRESS.search(txt)
    if m:
        address = _clean_address(m.group(1))
    if not address:
        m2 = RE_ADDRESS_FALLBACK.search(txt)
        if m2:
            cand = m2.group(1).strip()
            if "pamplona" in cand.lower() or "norte de santander" in cand.lower():
                address = _clean_address(cand)
    if not address:
        address = "Carrera 5 No. 5-88, Pamplona, Norte de Santander, Colombia."

    # Teléfonos
    phones_found = []
    for mm in PHONE_MOBILE.findall(txt):
        phones_found.append(_norm_phone(mm))
    for mm in PHONE_LAND.findall(txt):
        phones_found.append(_norm_phone(mm))
    phones, seen = [], set()
    for p in phones_found:
        if p and p not in seen:
            phones.append(p)
            seen.add(p)
    if not phones:
        phones = ["+57 333 033 3569", "(607) 568 0093"]

    # Correos
    emails = list({e.group(0).lower() for e in RE_EMAIL.finditer(txt)}) or [
        "ccpamplona@camarapamplona.org.co"
    ]

    # Horarios
    hours = _canon_hours(txt)
    return {"address": address, "phones": phones[:5], "emails": emails[:3], "hours": hours}

    # Correos
    emails = list({e.group(0).lower() for e in RE_EMAIL.finditer(txt)}) or ["ccpamplona@camarapamplona.org.co"]
    # Horarios
    hours = _canon_hours(txt)
    return {"address": address, "phones": phones[:5], "emails": emails[:3], "hours": hours}

def _extract_tarifas(text: str) -> List[str]:
    txt = _clean_text(text or "")
    parts = re.split(r"[•\-\u2022]\s*|[\n\r]+|(?<=\.)\s+", txt)
    out = []
    for p in parts:
        s = p.strip()
        if not s: continue
        if _TARIFA_KW.search(s) or _CURRENCY.search(s):
            out.append(_truncate(s, 180))
    seen, uniq = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            uniq.append(s); seen.add(k)
    return uniq[:8]

def _extract_tramites(text: str) -> List[str]:
    txt = _clean_text(text or "")
    parts = re.split(r"[•\-\u2022]\s*|[\n\r]+|(?<=\.)\s+", txt)
    out = []
    for p in parts:
        s = p.strip()
        if not s: continue
        if _REQ_KW.search(s):
            out.append(_truncate(s, 180))
    seen, uniq = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            uniq.append(s); seen.add(k)
    return uniq[:8]

# ---------------- Búsqueda en Qdrant ----------------
def _qdrant_search(query: str, limit: int = 32):
    try:
        qvec = hf_embed(query, kind="query")
    except Exception as e:
        print("[RAG] ERROR embed(query):", e)
        return [], []
    body = {"vector": qvec, "limit": limit, "with_payload": True, "with_vector": True, "score_threshold": 0.0}
    try:
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{QDRANT_URL}/collections/{COLL}/points/search", json=body)
            r.raise_for_status()
            hits = r.json().get("result", [])
    except Exception as e:
        print("[RAG] ERROR Qdrant search:", e)
        return [], qvec

    out = []
    for h in hits or []:
        p = h.get("payload", {}) or {}
        out.append({
            "text": p.get("text", "") or "",
            "path": p.get("path", "") or "",
            "filename": p.get("filename", "") or "",
            "chunk_index": p.get("chunk_index", 0),
            "vector": h.get("vector") or [],
            "score": float(h.get("score", 0.0)),
        })
    return out, qvec

# ---------------- Rerank ----------------
def _rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs: return []
    if _HAS_XENC and _XENC is not None:
        try:
            pairs = [[query, d["text"]] for d in docs]
            scores = _XENC.predict(pairs).tolist()
            for d, s in zip(docs, scores): d["score"] = float(s)
            docs.sort(key=lambda x: x["score"], reverse=True)
        except Exception as e:
            print("[RAG] ERROR CrossEncoder:", e)
    else:
        docs = sorted(docs, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return docs

# ---------------- Contexto para el LLM ----------------
def _build_context(docs: List[Dict[str, Any]], max_chars=1200) -> str:
    lines, used = [], 0
    for d in docs:
        chunk = re.sub(r"\s*\n\s*", " ", (d["text"] or "").strip())
        if not chunk: continue
        part = f"[{d['filename']}#{d['chunk_index']}] {chunk}"
        if len(part) > 360: part = part[:360].rsplit(" ", 1)[0]
        if used + len(part) > max_chars: break
        lines.append(part); used += len(part)
    return "\n".join(lines)

# ---------------- Post-procesado de respuesta ----------------
def _ensure_sentence_complete(text: str) -> str:
    text = (text or "").strip()
    if not text: return text
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last != -1 and last >= len(text) - 4: return text
    if last != -1: return text[: last + 1].strip()
    return text

def _limit_section_sentences(ans: str, max_sent: int = 3) -> str:
    if not ans: return ans
    lines = ans.splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if s.startswith("**"):  # encabezado
            out.append(s)
            continue
        piezas = re.split(r"(?<=[\.\!\?])\s+", s)
        piezas = [p.strip() for p in piezas if p.strip()]
        out.append(" ".join(piezas[:max_sent]))
    return "\n".join(out)

def _hard_trim(ans: str, max_chars: int = 600) -> str:
    ans = re.sub(r"\s{2,}", " ", ans or "").strip()
    if len(ans) <= max_chars:
        return ans
    cut = ans[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"

def _sanitize_output(ans: str) -> str:
    if not ans: return ans
    # quitar líneas residuales de ruido
    ans = re.sub(RE_NOISE, "", ans)
    # compactar bullets
    ans = re.sub(r"\n{3,}", "\n\n", ans)
    # limitar por sección y total
    ans = _limit_section_sentences(ans, max_sent=3)
    ans = _hard_trim(ans, max_chars=600)
    return ans

# ---------------- LLM ----------------
def _llm_answer(prompt: str) -> str:
    if not GROQ_KEY:
        return ""
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 360,
        "top_p": 1.0,
    }
    try:
        with httpx.Client(timeout=55) as c:
            r = c.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
            if r.status_code != 200:
                print("[RAG] ERROR LLM:", r.status_code, r.text[:200])
                return ""
            data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return _ensure_sentence_complete(text)
    except Exception as e:
        print("[RAG] ERROR LLM:", e)
        return ""

# ---------------- Respuesta principal ----------------
def answer_with_rag(query: str) -> str:
    q = query or ""
    # Small talk
    st = _smalltalk_reply(q)
    if st is not None:
        return st

    intents = _detect_intents_mixed(q)

    # Buscar en Qdrant
    hits, _ = _qdrant_search(q, limit=32)
    if not hits:
        info = _extract_contact_block("")
        parts = [f"**Dirección:** {info['address']}",
                 f"**Teléfonos/WhatsApp:** {', '.join(info['phones'])}",
                 f"**Correo:** {', '.join(info['emails'])}"]
        if "horarios" in intents:
            parts.append("**Horarios:** Lunes a viernes: 8:00 a. m.–12:00 m. y 2:00 p. m.–6:00 p. m.")
        parts.append("¿Te indico *Trámites* o *Certificados*?")
        return _sanitize_output("\n".join(parts))

    # Reordenar y seleccionar
    hits = _rerank(q, hits)[:16]
    top = _mmr_select(hits, k=6)

    joined_raw = " \n ".join(d.get("text", "") for d in top)
    low = (joined_raw or "").lower()
    is_pamplona = ("pamplona" in low) or ("norte de santander" in low) or ("norte de santandér" in low)
    has_other_cities = bool(CITY_BLOCKLIST.search(low))
    if not is_pamplona or has_other_cities:
        joined = "Dirección: Carrera 5 No. 5-88, Pamplona, Norte de Santander. Teléfonos: +57 333 033 3569, (607) 568 0093. Correo: ccpamplona@camarapamplona.org.co."
    else:
        joined = joined_raw

    # Determinista para operativas
    if any(x in intents for x in ("ubicacion","horarios","telefonos")):
        info = _extract_contact_block(joined)
        out = []
        if "ubicacion" in intents:
            out.append(f"**Dirección:** {info['address']}")
        if "horarios" in intents:
            hrs = info["hours"] or ["Lunes a viernes: 8:00 a. m.–12:00 m. y 2:00 p. m.–6:00 p. m."]
            out.append("**Horarios:** " + " | ".join(hrs[:2]))
        if "telefonos" in intents:
            out.append("**Teléfonos/WhatsApp:** " + ", ".join(info["phones"]))
            out.append("**Correo:** " + ", ".join(info["emails"]))
        if out:
            out.append("¿Necesitas *Trámites* o *Tarifas*?")
            return _sanitize_output("\n".join(out))

    # Institucional (mision/vision) → LLM con contexto limitado
    if intents == ["mision_vision"]:
        ctx = _build_context(top)
        prompt = (
            "Del CONTEXTO devuelve solo Misión y/o Visión (y Valores/Política de calidad si aparecen). "
            "Sin otros temas. Máx. 3 oraciones por sección.\n\n"
            f"CONTEXTO:\n{ctx}\n\nPregunta: {q}"
        )
        ans = _llm_answer(prompt)
        if ans:
            return _sanitize_output(ans + "\n¿Deseas *Trámites* o *Horarios*?")

    if intents == ["tramites"] or intents == ["conciliacion"]:
        items = _extract_tramites(joined)
        if items:
            titulo = "**Conciliación – requisitos (resumen):**" if intents == ["conciliacion"] else "**Trámites (resumen):**"
            return _sanitize_output(f"{titulo}\n- " + "\n- ".join(items) + "\n¿Te detallo algún *requisito* específico?")

    if intents == ["tarifas"]:
        items = _extract_tarifas(joined)
        if items:
            return _sanitize_output("**Tarifas / costos:**\n- " + "\n- ".join(items) + "\n¿Buscas *certificados* o *pagos*?")

    # Respaldo con LLM (breve)
    ctx = _build_context(top)
    prompt = (
        "Responde SOLO con información de la Cámara de Comercio de Pamplona (Norte de Santander). "
        "Sé breve y exacto; máx. 3 oraciones por sección. Si faltan datos, indícalo sin inventar.\n\n"
        f"CONTEXTO:\n{ctx}\n\nPregunta: {q}"
    )
    llm_out = _llm_answer(prompt)
    if llm_out:
        return _sanitize_output(llm_out)

    return "No encuentro información sobre esa consulta. ¿Quieres que te indique *Ubicación*, *Horarios* o *Trámites*?"
