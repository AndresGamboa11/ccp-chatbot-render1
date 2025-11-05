# app/whatsapp.py
import httpx
from app.settings import get_settings

S = get_settings()

# Límite máximo de caracteres permitido por WhatsApp
WA_LIMIT = 4096


def split_for_whatsapp(text: str, limit: int = WA_LIMIT) -> list[str]:
    """
    Divide el texto largo en partes sin cortar palabras.
    """
    out, cur = [], text.strip()
    while len(cur) > limit:
        # intenta cortar en el último salto o punto
        cut = max(
            cur.rfind("\n\n", 0, limit),
            cur.rfind(". ", 0, limit),
            cur.rfind("\n", 0, limit),
            cur.rfind(" ", 0, limit),
        )
        if cut < int(limit * 0.6):  # si no hay buen corte
            cut = limit
        out.append(cur[:cut].strip())
        cur = cur[cut:].lstrip()
    if cur:
        out.append(cur)
    return out


async def send_whatsapp_text(to_number: str, body: str) -> dict:
    """
    Envía texto por WhatsApp Cloud API usando los valores del entorno.
    """
    url = f"https://graph.facebook.com/{S.WA_API_VERSION}/{S.WA_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {S.WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    parts = split_for_whatsapp(body)
    last_resp = {}

    async with httpx.AsyncClient(timeout=60) as client:
        for p in parts:
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "text",
                "text": {"body": p},
            }
            resp = await client.post(url, headers=headers, json=payload)
            try:
                last_resp = {"status_code": resp.status_code, **resp.json()}
            except Exception:
                last_resp = {"status_code": resp.status_code, "text": resp.text}

            # Si hay error (401/403/400), deja de enviar más partes
            if not (200 <= resp.status_code < 300):
                print("[WA SEND ERROR]", resp.text[:200])
                break

    return last_resp
