import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

_SYSTEM_PROMPT = """Eres un traductor especializado de Lengua de Signos Española (LSE) a español escrito.

TAREA: Recibes una lista de glosas en mayúsculas (palabras sueltas en orden LSE) y debes convertirlas en UNA frase en español natural y gramaticalmente correcta.

REGLAS OBLIGATORIAS:
1. REORDENA las palabras al orden natural del español (sujeto-verbo-objeto).
2. AÑADE los verbos auxiliares que falten: "YO BIEN" → "Yo estoy bien.", "YO HAMBRE" → "Yo tengo hambre."
3. AÑADE artículos, preposiciones y nexos necesarios.
4. Si las glosas forman una pregunta (hay CUAL, DONDE, CUANDO, COMO, QUIEN, CUANTO), añade ¿ y ?.
5. Devuelve ÚNICAMENTE la frase final. Sin explicaciones, sin notas, sin comillas.

EJEMPLOS:
- TU, NOMBRE, CUAL → ¿Cuál es tu nombre?
- YO, BIEN → Yo estoy bien.
- YO, HAMBRE → Yo tengo hambre.
- TU, DONDE, VIVIR → ¿Dónde vives?
- ADIOS, GRACIAS → Adiós, muchas gracias.
- YO, LLAMAR, PABLO → Me llamo Pablo.
- TU, CUANTOS, AÑOS → ¿Cuántos años tienes?"""

# Buffer de palabras detectadas
buffer = []
last_word = None


def add_word(word: str):
    # Guarda la palabra detectada en el buffer evitando duplicados inmediatos
    global last_word, buffer
    word = word.upper().strip()

    if word != last_word:
        buffer.append(word)
        last_word = word
        print(f"[Buffer] Palabra guardada: {word} | Buffer actual: {buffer}")


def translate_current_buffer(lista_glosas: list) -> str:
    # Traduce todo lo acumulado de un usuario, limpia su buffer local y devuelve la frase.
    if not lista_glosas:
        return ""

    # Copiamos las glosas actuales para procesar
    glosses_to_translate = lista_glosas.copy()
    
    # Al ser las listas objetos mutables pasados por referencia, 
    # este clear() vacía directamente el buffer del usuario dentro de client_data[sid]
    lista_glosas.clear()

    gloss_str = ", ".join(glosses_to_translate)
    print(f"[Ollama] Traduciendo: {glosses_to_translate}")

    prompt = f"Glosas LSE a traducir: {gloss_str}"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "system": _SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 80,
                }
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if 'total_duration' in data:
            milisegundos = data['total_duration'] / 1_000_000
            print(f"[Ollama] Inferencia completada en: {milisegundos:.2f} ms")

        translated = data["response"].strip()
        print(f"[Ollama] Traducción: '{translated}'")
        return translated

    except Exception as e:
        print(f"[Error Ollama]: {e}")
        fallback = f"{' '.join(glosses_to_translate).lower().capitalize()}."
        print(f"[Ollama] Fallback (Sin IA): '{fallback}'")
        return fallback