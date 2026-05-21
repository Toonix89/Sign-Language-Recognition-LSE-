# -*- coding: utf-8 -*-
"""
sign_buffer_manager.py
======================
SLT (Sign Language Translation) pipeline - Buffer layer.

Architecture position:
    [BiLSTM Vision Model] -> SignBufferManager -> [LLM API] -> [TTS]

This script can run standalone as a simulator to test buffer logic and
LLM prompts independently from the vision model.
"""

import asyncio
import os
import sys
import time
from typing import Callable, Awaitable

import google.generativeai as genai

# Force UTF-8 output on Windows terminals (PowerShell / cmd with cp1252)
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# SECTION 1 - Real Gemini Flash API
# ==============================================================================

# Load API key from environment variable (recommended) or hardcode for quick testing.
# Set the env var with:  $env:GEMINI_API_KEY="AIzaSy..."
_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. "
        "Run: $env:GEMINI_API_KEY='YOUR_KEY' before launching."
    )
genai.configure(api_key=_api_key)

# Instantiate the model ONCE at module level - not on every translation call.
_SYSTEM_PROMPT = (
    "Eres el motor de traduccion (SLT) de un sistema de Lengua de Signos Espanola (LSE). "
    "Tu tarea es recibir una lista de glosas en mayusculas y transformarlas en una frase "
    "en espanol que sea natural, fluida y gramaticalmente correcta.\n\n"
    "Reglas estrictas:\n"
    "1. Anade los articulos, preposiciones y verbos auxiliares (ser/estar) que falten.\n"
    "2. Conjuga correctamente los verbos segun el contexto de la frase.\n"
    "3. Devuelve UNICAMENTE la frase final traducida. No incluyas introducciones, "
    "explicaciones ni notas (ej. No pongas 'Aqui tienes la traduccion:')."
)

_gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=_SYSTEM_PROMPT,
)


async def call_llm_api(glosses: list[str]) -> str:
    """
    Calls Gemini Flash asynchronously to translate an LSE gloss sequence
    into a grammatically correct Spanish sentence.

    Parameters
    ----------
    glosses : list[str]
        Ordered list of LSE glosses from the buffer, e.g. ["YO", "ALTO"].

    Returns
    -------
    str
        A grammatically correct Spanish sentence, or a fallback string on error.
    """
    print(f"\n  +--[LLM API]------------------------------------------")
    print(f"  |  Glosses sent  : {glosses}")

    # Format glosses as a clean comma-separated string for the prompt
    prompt = f"Glosas a traducir: {', '.join(glosses)}"

    try:
        # Non-blocking call - the asyncio event loop stays fully responsive
        response = await _gemini_model.generate_content_async(prompt)
        sentence = response.text.strip()

        print(f"  |  Response      : \"{sentence}\"")
        print(f"  +-----------------------------------------------------")
        return sentence

    except Exception as e:
        print(f"  |  Gemini error  : {e}")
        print(f"  +-----------------------------------------------------")
        # Fallback: concatenate glosses so TTS is never left completely silent
        return f"{' '.join(glosses).lower().capitalize()}."


# ==============================================================================
# SECTION 2 - SignBufferManager
# ==============================================================================

class SignBufferManager:
    """
    Manages the gloss buffer between the vision model and the NLP layer.

    Responsibilities
    ----------------
    - Accumulates gloss words detected by the vision model.
    - Ignores consecutive duplicate words (model "stuck" on the same sign).
    - Maintains a restartable timeout; when it fires, calls trigger_translation().
    - Delegates sentence construction to an async LLM function.

    Parameters
    ----------
    timeout : float
        Seconds of silence after the last word before translation is triggered.
        Default: 1.5 s. Tune this value in the simulator first.
    on_sentence : Callable[[str], Awaitable[None]] | None
        Optional async callback invoked with the final Spanish sentence.
        Use this to pipe the output to a TTS engine.
    """

    def __init__(
        self,
        timeout: float = 1.5,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.timeout = timeout
        self.on_sentence = on_sentence

        # The gloss buffer (cleared after each successful translation)
        self.buffer: list[str] = []

        # Track the last accepted word to filter consecutive duplicates
        self._last_word: str | None = None

        # Handle to the running asyncio timeout task (None if no words yet)
        self._timer_task: asyncio.Task | None = None

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def add_word(self, word: str) -> None:
        """
        Called by the vision model each time a sign is confirmed.

        Consecutive duplicates are silently dropped so that a model stuck
        on the same prediction for several frames does not flood the buffer.

        Parameters
        ----------
        word : str
            Uppercase gloss string, e.g. "HOLA".
        """
        word = word.upper().strip()

        if word == self._last_word:
            # Same word as before - skip without resetting the timer
            return

        self._last_word = word
        self.buffer.append(word)

        elapsed = time.monotonic()
        print(f"  [Buffer +] '{word}' added -> {self.buffer}  (t={elapsed:.2f}s)")

        # Every new, distinct word restarts the timeout countdown
        self._restart_timer()

    async def trigger_translation(self) -> str | None:
        """
        Sends the current buffer to the LLM and clears it.
        Can be called manually (e.g. by a 'done' gesture) or automatically
        by the internal timeout.

        Returns the translated sentence, or None if the buffer was empty.
        """
        if not self.buffer:
            print("  [Buffer] Timeout fired but buffer is empty - nothing to translate.")
            return None

        # Snapshot and clear the buffer before the async call to avoid
        # race conditions if add_word() is called during the LLM await.
        glosses = self.buffer.copy()
        self._clear_buffer()

        print(f"\n  [Timeout] Silence detected - translating buffer: {glosses}")

        sentence = await call_llm_api(glosses)

        # Forward to TTS layer if a callback was registered
        if self.on_sentence:
            await self.on_sentence(sentence)

        return sentence

    def force_translate_now(self) -> None:
        """
        Convenience helper: schedules an immediate translation from
        synchronous code (e.g. a button press or a 'done' gesture handler).
        Cancels any pending timeout first.
        """
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        asyncio.ensure_future(self.trigger_translation())

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _restart_timer(self) -> None:
        """Cancels any running timer and starts a fresh one."""
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()

        # create_task() schedules the coroutine on the running event loop
        # without blocking the caller - the event loop stays fully responsive.
        self._timer_task = asyncio.create_task(self._timeout_handler())

    async def _timeout_handler(self) -> None:
        """
        Waits `self.timeout` seconds then fires the translation.
        If cancelled (because a new word arrived), exits silently.
        """
        try:
            await asyncio.sleep(self.timeout)
            await self.trigger_translation()
        except asyncio.CancelledError:
            # Timer was reset by a new incoming word - expected, not an error.
            pass

    def _clear_buffer(self) -> None:
        """Resets buffer and duplicate-guard so the next phrase starts fresh."""
        self.buffer.clear()
        self._last_word = None
        self._timer_task = None


# ==============================================================================
# SECTION 3 - TTS stub callback
# Replace with a real call: pyttsx3, gTTS, or Web Speech API via Socket.IO
# ==============================================================================

async def speak(sentence: str) -> None:
    """Stub TTS callback. Replace with actual speech synthesis."""
    print(f"\n  [TTS] Speaking: \"{sentence}\"\n")


# ==============================================================================
# SECTION 4 - Simulation
# Mimics the vision model sending words at irregular intervals.
# Run:  python sign_buffer_manager.py
# ==============================================================================

async def run_simulation() -> None:
    """
    Simulates the vision model emitting signs at irregular intervals.

    Timeline
    --------
    t=0.0s  -> "YO"    arrives (added)
    t=0.5s  -> "YO"    arrives again (DUPLICATE - ignored)
    t=0.8s  -> "ALTO"  arrives (added)
    t=1.0s  -> "ALTO"  arrives again (DUPLICATE - ignored)
    t=2.3s  -> 1.5 s silence -> timeout fires -> translate ["YO", "ALTO"]
    t=4.3s  -> "HOLA"  starts a new phrase
    t=4.8s  -> "YO"    arrives
    t=5.2s  -> "SORDO" arrives
    t=6.7s  -> timeout fires -> translate ["HOLA", "YO", "SORDO"]
    """

    print("=" * 60)
    print("  SignBufferManager - Simulation")
    print(f"  Timeout: 1.5 s  |  Start: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    manager = SignBufferManager(timeout=1.5, on_sentence=speak)

    # --- Phrase 1: "YO ALTO" --------------------------------------------------
    print("\n--- Phrase 1: 'YO ALTO' --------------------------------")

    await asyncio.sleep(0.0)
    manager.add_word("YO")            # t=0.0 -- added

    await asyncio.sleep(0.5)
    manager.add_word("YO")            # t=0.5 -- DUPLICATE, ignored

    await asyncio.sleep(0.3)
    manager.add_word("ALTO")          # t=0.8 -- added

    await asyncio.sleep(0.2)
    manager.add_word("ALTO")          # t=1.0 -- DUPLICATE, ignored

    # 1.5 s of silence will trigger translation
    print("  [Sim] Silence... waiting for timeout")
    await asyncio.sleep(1.8)          # wait past the 1.5 s threshold

    # --- Phrase 2: "HOLA YO SORDO" -------------------------------------------
    print("\n--- Phrase 2: 'HOLA YO SORDO' -------------------------")

    manager.add_word("HOLA")
    await asyncio.sleep(0.5)
    manager.add_word("YO")
    await asyncio.sleep(0.4)
    manager.add_word("SORDO")

    print("  [Sim] Silence... waiting for timeout")
    await asyncio.sleep(2.0)          # timeout fires at +1.5 s

    # --- Done -----------------------------------------------------------------
    print("=" * 60)
    print("  Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_simulation())
