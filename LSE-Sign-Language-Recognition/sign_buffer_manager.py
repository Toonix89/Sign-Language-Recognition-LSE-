import asyncio
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Cuando detecta palabra se crashea

# Configuración de codificación para evitar problemas en Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Carga de la API Key de Gemini desde el archivo .env
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(parent_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError("No se encontró la clave GEMINI_API_KEY en el archivo .env")

genai.configure(api_key=_api_key)

# Prompt para el comportamiento de Gemini
_SYSTEM_PROMPT = (
    "Eres el motor de traducción (SLT) de un sistema de Lengua de Signos Española (LSE).\n"
    "Tu tarea es recibir una lista de glosas en mayúsculas y transformarlas en una frase "
    "en español que sea natural, fluida y gramaticalmente correcta.\n\n"
    "Reglas estrictas:\n"
    "1. Añade los artículos, preposiciones y verbos auxiliares (ser/estar) que falten.\n"
    "2. Conjuga correctamente los verbos según el contexto de la frase.\n"
    "3. Devuelve ÚNICAMENTE la frase final traducida. No incluyas explicaciones ni notas."
)

_gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=_SYSTEM_PROMPT,
)


# =====================================================================
# VARIABLES GLOBALES (Sustituyen a la clase/objeto)
# =====================================================================
buffer = []                # Lista donde se guardan las palabras
last_word = None           # Guarda la última palabra para evitar repetidas
timer_task = None          # Tarea que controla el tiempo en segundo plano
TIMEOUT_SECONDS = 1.5      # Segundos de silencio para traducir


# =====================================================================
# FUNCIONES DEL SISTEMA
# =====================================================================

async def translate_glosses(glosses):
    """Envía las glosas a Gemini y devuelve la frase traducida"""
    if not glosses:
        return ""

    prompt = f"Glosas a traducir: {', '.join(glosses)}"

    try:
        response = await _gemini_model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Error LLM]: {e}")
        return f"{' '.join(glosses).lower().capitalize()}."


def add_word(word):
    """Añade una palabra al búfer global evitando duplicados rápidos"""
    global last_word, buffer
    
    word = word.upper().strip()

    # Si es igual a la última palabra, no hacemos nada y salimos de la función
    if word == last_word:
        return

    last_word = word
    buffer.append(word)

    # Cada vez que entra una palabra nueva, reiniciamos el temporizador
    _restart_timer()


async def trigger_translation():
    """Vacía el búfer y pide la traducción"""
    global buffer
    
    if not buffer:
        return

    # Copiamos las palabras acumuladas y limpiamos el almacén
    glosses_to_translate = buffer.copy()
    _clear_buffer()

    print(f"\n[Traduciendo] Enviando a Gemini: {glosses_to_translate}")
    
    # Llamamos a la IA y esperamos la respuesta
    frase_final = await translate_glosses(glosses_to_translate)
    
    # Aquí puedes hacer lo que quieras con la frase (imprimirla, reproducir voz, etc.)
    print(f"[Resultado]: {frase_final}\n")


def _restart_timer():
    """Cancela el temporizador anterior y arranca uno nuevo"""
    global timer_task
    
    # Si ya había un temporizador contando, lo paramos
    if timer_task and not timer_task.done():
        timer_task.cancel()

    # Arrancamos una cuenta atrás nueva en segundo plano
    timer_task = asyncio.create_task(_timeout_handler())


async def _timeout_handler():
    """Espera el tiempo de silencio y si nadie lo cancela, traduce"""
    global TIMEOUT_SECONDS
    try:
        await asyncio.sleep(TIMEOUT_SECONDS)
        await trigger_translation()
    except asyncio.CancelledError:
        # Si saltó esta excepción es porque llegó otra palabra y cancelamos el tiempo
        pass


def _clear_buffer():
    """Limpia las variables para la siguiente frase"""
    global buffer, last_word, timer_task
    buffer.clear()
    last_word = None
    timer_task = None
