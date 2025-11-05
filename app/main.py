# app/main.py
import inspect
from pathlib import Path
from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.rag import answer_with_rag
from app.settings import get_settings
from app.whatsapp import send_whatsapp_text

# -------------------------------------------------------------------
# Configuración base
# -------------------------------------------------------------------
S = get_settings()
app = FastAPI(title="Chatbot CCP")

# Carpeta estática
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------------------------------------
# RESPUESTAS RÁPIDAS (SALUDOS Y FRASES COTIDIANAS)
# ----------------------------------------------------------
COMMON_RESPONSES = {
    "hola": "¡Hola! 👋 Soy el asistente virtual de la Cámara de Comercio de Pamplona. ¿En qué puedo ayudarte hoy?",
    "buenos dias": "¡Buenos días! ☀️ ¿Deseas información sobre algún trámite o servicio de la CCP?",
    "buenos días": "¡Buenos días! ☀️ ¿Deseas información sobre algún trámite o servicio de la CCP?",
    "buenas tardes": "¡Buenas tardes! 🌞 Estoy aquí para ayudarte con información de la Cámara de Comercio de Pamplona.",
    "buenas noches": "¡Buenas noches! 🌙 Si necesitas información sobre los servicios o trámites de la Cámara, con gusto te ayudo.",
    "como estas": "Estoy muy bien, ¡gracias por preguntar! 😊 ¿En qué puedo ayudarte hoy?",
    "cómo estás": "Estoy muy bien, ¡gracias por preguntar! 😊 ¿En qué puedo ayudarte hoy?",
    "gracias": "¡Con gusto! 😊 Si necesitas más información sobre la Cámara, estaré aquí para ayudarte.",
    "muchas gracias": "¡Con gusto! 😊 ¿Deseas consultar algún trámite o certificación?",
    "adios": "¡Hasta pronto! 👋 Recuerda que puedes escribirme cuando necesites información de la CCP.",
    "adiós": "¡Hasta pronto! 👋 Recuerda que puedes escribirme cuando necesites información de la CCP.",
}

# Sugerencia corta cuando es solo saludo o texto muy breve
DEFAULT_GREETING_MENU = (
    "¡Hola! Soy el asistente de la Cámara de Comercio de Pamplona. "
    "Puedo ayudarte con:\n"
    "• Horarios de atención\n"
    "• Trámites del registro mercantil\n"
    "• Certificados y costos\n"
    "• Dirección y contacto\n\n"
    "Pregúntame, por ejemplo: *¿Cuáles son los horarios de atención?*"
)

def detect_common_response(text: str):
    """
    Devuelve una respuesta rápida si el texto contiene un patrón cotidiano.
    Revisa por inclusión ('key in text') para cubrir variantes.
    """
    t = text.lower().strip()
    for key, resp in COMMON_RESPONSES.items():
        if key in t:
            return resp
    return None


# -------------------------------------------------------------------
# Rutas básicas
# -------------------------------------------------------------------
@app.get("/")
def home():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Chatbot CCP online ✅ (coloca tu index.html en /static)"}


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------------------------------------------------
# Endpoint de prueba manual del RAG
# -------------------------------------------------------------------
@app.get("/ask_test", tags=["debug"])
async def ask_test(q: str = Query(..., min_length=1)):
    """
    Permite probar el modelo SIN pasar por WhatsApp.
    Primero intenta respuestas rápidas; si no, usa RAG.
    """
    quick = detect_common_response(q)
    if quick:
        return {"q": q, "answer": quick}

    if len(q.strip()) <= 3:
        return {"q": q, "answer": DEFAULT_GREETING_MENU}

    if inspect.iscoroutinefunction(answer_with_rag):
        ans = await answer_with_rag(q)
    else:
        ans = answer_with_rag(q)
    return {"q": q, "answer": ans}


# -------------------------------------------------------------------
# Normalizador del texto recibido desde WhatsApp
# -------------------------------------------------------------------
def extract_wa_text(entry: dict) -> str:
    """
    Soporta text, button, interactive (list/button), y media con caption.
    Devuelve siempre un string (o '' si no hay texto válido).
    """
    try:
        msg = entry["entry"][0]["changes"][0]["value"]["messages"][0]
    except Exception:
        return ""

    t = msg.get("type")

    if t == "text":
        return (msg.get("text", {}).get("body") or "").strip()

    if t == "button":
        return (msg.get("button", {}).get("text") or "").strip()

    if t == "interactive":
        inter = msg.get("interactive", {})
        # list reply
        if inter.get("type") == "list_reply" or inter.get("list_reply"):
            lr = inter.get("list_reply") or inter.get("list", {}).get("reply") or {}
            return (lr.get("title") or lr.get("id") or "").strip()
        # button reply
        if inter.get("type") == "button_reply" or inter.get("button_reply"):
            br = inter.get("button_reply") or {}
            return (br.get("title") or br.get("id") or "").strip()

    # Media con caption
    for k in ("image", "document", "video", "audio"):
        if k in msg:
            return (msg.get(k, {}).get("caption") or f"[{k} sin texto]").strip()

    return ""


# -------------------------------------------------------------------
# Webhook WhatsApp - verificación (GET)
# -------------------------------------------------------------------
@app.get("/webhook")
async def webhook_verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == (S.WA_VERIFY_TOKEN or ""):
        # WhatsApp requiere texto plano
        return PlainTextResponse(str(challenge))

    return JSONResponse({"error": "verify token inválido"}, status_code=403)


# -------------------------------------------------------------------
# Webhook WhatsApp - recepción de mensajes (POST)
# -------------------------------------------------------------------
@app.post("/webhook")
async def webhook_receive(req: Request):
    try:
        data = await req.json()
    except Exception as e:
        print("[WEBHOOK] ERROR leyendo JSON:", e)
        return {"ok": False, "error": "bad json"}

    # 1) Filtrar eventos que no son mensajes (ej. statuses / message_deliveries)
    try:
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])
        if not messages:
            # Nada que procesar (pudo ser un 'status' u otro webhook)
            return {"ok": True, "note": "no message in payload"}
        msg = messages[0]
        from_id = msg.get("from")
        if not from_id:
            return {"ok": True, "note": "no sender"}
    except Exception as e:
        print("[WEBHOOK] ERROR extrayendo mensaje:", e)
        return {"ok": True, "note": "malformed payload"}

    # 2) Extraer el texto de manera segura
    text = extract_wa_text(data).strip()
    if not text:
        await send_whatsapp_text(
            from_id,
            "No recibí texto. Por favor envíame tu consulta en un mensaje de texto."
        )
        return {"ok": True, "note": "empty text"}

    # 3) Intentar respuestas rápidas (no pasar por RAG)
    quick = detect_common_response(text)
    if quick:
        await send_whatsapp_text(from_id, quick)
        return {"ok": True, "handled": "quick_reply"}

    # Si el mensaje es muy corto (ej. “ok”, “hi”, “?”), enviar menú por defecto
    if len(text) <= 3:
        await send_whatsapp_text(from_id, DEFAULT_GREETING_MENU)
        return {"ok": True, "handled": "short_text"}

    # 4) Llamar a RAG (con try/except para no romper el flujo)
    try:
        if inspect.iscoroutinefunction(answer_with_rag):
            ans = await answer_with_rag(text)
        else:
            ans = answer_with_rag(text)
    except Exception as e:
        # Respuesta amable + log detallado
        print("[RAG ERROR]", e)
        import traceback; print(traceback.format_exc())
        ans = (
            "Tuve un problema procesando la consulta. "
            "Intenta de nuevo con otra redacción, por ejemplo: *¿Cuáles son los horarios de atención?*"
        )

    # 5) Enviar respuesta a WhatsApp (el helper ya divide mensajes largos)
    result = await send_whatsapp_text(from_id, ans)

    # 6) Log básico del envío (se adapta al dict que devuelve send_whatsapp_text)
    if not result.get("ok", True):
        print("[WA SEND ERROR]", result)

    return {"ok": True, "send": result}


# -------------------------------------------------------------------
# Diagnóstico de conexión con Groq
# -------------------------------------------------------------------
@app.get("/diag/groq")
def diag_groq():
    try:
        from app.rag import _llm_answer
        out = _llm_answer("Di 'pong' si estás operativo.")
        return {"ok": True, "sample": out[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}
