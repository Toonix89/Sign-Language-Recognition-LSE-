import threading
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

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
# VARIABLES GLOBALES Y CERROJOS (Seguras para hilos)
# =====================================================================
buffer = []                     # Lista donde se guardan las palabras
last_word = None                # Guarda la última palabra para evitar repetidas
timer_thread = None             # Objeto threading.Timer que controla el tiempo en segundo plano
TIMEOUT_SECONDS = 1.5           # Segundos de silencio para traducir
buffer_lock = threading.Lock()  # Candado de seguridad para evitar que la cámara y el temporizador choquen
callback_on_translation = None  # Callback para notificar al frontend vía Socket.IO (enviar_traduccion_a_frontend de server.py)


# =====================================================================
# FUNCIONES DEL SISTEMA
# =====================================================================

def translate_glosses(glosses):
    """Envía las glosas a Gemini de forma síncrona (corre en hilo secundario)"""
    if not glosses:
        return ""

    prompt = f"Glosas a traducir: {', '.join(glosses)}"

    try:
        # Usamos generate_content (SÍNCRONO). No bloquea la cámara porque corre en otro hilo
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Error LLM]: {e}")
        return f"{' '.join(glosses).lower().capitalize()}."


def add_word(word):
    """Añade una palabra al búfer protegiendo los datos con el cerrojo"""
    global last_word, buffer
    
    word = word.upper().strip()

    # Ponemos el candado mientras tocamos el buffer y reactivamos el tiempo
    with buffer_lock:
        if word == last_word:
            return

        last_word = word
        buffer.append(word)
        _restart_timer()


def trigger_translation():
    """Vacía el búfer de forma segura y lanza la petición a Gemini"""
    global buffer, callback_on_translation
    
    # Entramos un momento a obtener los datos del buffer y limpiarlo de forma segura
    with buffer_lock:
        if not buffer:
            return
        glosses_to_translate = buffer.copy()
        _clear_buffer()

    # El resto de la función corre FUERA del candado para no ralentizar el sistema
    print(f"\n[Traduciendo] Enviando a Gemini: {glosses_to_translate}")
    frase_final = translate_glosses(glosses_to_translate)
    print(f"[Resultado]: {frase_final}\n")

    # Invocamos el callback para notificar al frontend vía Socket.IO
    if callback_on_translation is not None:
        try:
            callback_on_translation(frase_final)
        except Exception as e:
            print(f"[Error Callback]: {e}")


def _restart_timer():
    """Cancela el temporizador de hilos anterior y arranca uno nuevo (Debe llamarse bajo lock)"""
    global timer_thread
    
    # Si el hilo del temporizador está vivo esperando el segundo, lo cancelamos
    if timer_thread is not None:
        timer_thread.cancel()

    # Creamos un temporizador que ejecutará 'trigger_translation' tras 1.5 segundos
    timer_thread = threading.Timer(TIMEOUT_SECONDS, trigger_translation)
    timer_thread.start()


def _clear_buffer():
    """Limpia las variables (Se ejecuta siempre dentro de un bloque con candado)"""
    global buffer, last_word, timer_thread
    buffer.clear()
    last_word = None
    timer_thread = None
