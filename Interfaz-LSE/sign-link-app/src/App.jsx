import React, { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Video, VideoOff, Settings, Hand } from 'lucide-react';
import { io } from 'socket.io-client';

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

if (recognition) {
  recognition.continuous = true; // Para que no se pare al revés de la primera pausa
  recognition.interimResults = true; // Para ver las palabras en pantalla mientras hablas
  recognition.lang = 'es-ES'; // Idioma configurado en castellano
}

function App() {
  const videoRef = useRef(null);                             // Referencia al elemento de vídeo
  const [showSettings, setShowSettings] = useState(false);    // Panel de configuración
  const [isMuted, setIsMuted] = useState(true);               // Estado del micrófono (Inicia silenciado por seguridad)
  const [isCameraOff, setIsCameraOff] = useState(false);      // Estado de la cámara 
  const [isAiActive, setIsAiActive] = useState(false);        // Estado de la IA de signos
  const [isAiVoiceActive, setIsAiVoiceActive] = useState(false); // Estado de la voz automática
  const [selectedVoice, setSelectedVoice] = useState('male'); // Género de la voz elegida
  const [showMesh, setShowMesh] = useState(true);             // Visualizar malla de MediaPipe
  const [showConfidence, setShowConfidence] = useState(true); // Mostrar % de precisión
  const [showSubtitle, setShowSubtitle] = useState(true);     // Mostrar subtítulos en pantalla
  const [confidence, setConfidence] = useState(0);           // Valor de precisión recibido
  const [subtitle, setSubtitle] = useState("...");            // Texto dinámico de los subtítulos
  const [landmarks, setLandmarks] = useState([]);              // Puntos clave de las manos
  const [sentence, setSentence] = useState('');               // Frase final traducida acumulada
  const [wordBuffer, setWordBuffer] = useState([]);           // Búfer de palabras detectadas
  const [isTranslating, setIsTranslating] = useState(false);  // Estado de traducción

  const canvasRef = useRef(null);                             // Canvas oculto para extraer fotogramas
  const overlayRef = useRef(null);                            // Canvas visible para dibujar los puntos
  const socketRef = useRef(null);                             // Instancia de Socket.IO
  const isProcessingRef = useRef(false);                      // Flag de control de flujo (Throttling)

  const isAiVoiceActiveRef = useRef(isAiVoiceActive);
  const selectedVoiceRef = useRef(selectedVoice);

  useEffect(() => {
    isAiVoiceActiveRef.current = isAiVoiceActive;
  }, [isAiVoiceActive]);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  // Inicializar webcam pidiendo permisos de VÍDEO y AUDIO simultáneamente
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error al acceder a los periféricos multimedia:', error);
      }
    };

    startVideo();
  }, []);

  // Google text to speech
  const synthesizeAndPlaySpeech = async (textToSpeak, voiceGender) => {
    if (!textToSpeak.trim()) return;

    const voiceName = voiceGender === 'female' ? 'es-ES-Neural2-A' : 'es-ES-Neural2-B';

    try {
      const response = await fetch(
        `https://texttospeech.googleapis.com/v1/text:synthesize?key=${import.meta.env.VITE_GOOGLE_TTS_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input: { text: textToSpeak },
            voice: { languageCode: 'es-ES', name: voiceName },
            audioConfig: { audioEncoding: 'MP3' }
          })
        }
      );

      const data = await response.json();
      if (data.audioContent) {
        const audio = new Audio(`data:audio/mp3;base64,${data.audioContent}`);
        audio.play();
      }
    } catch (err) {
      console.error("Error en el servicio de síntesis Google TTS:", err);
    }
  };

  // Socket.io
  useEffect(() => {
    socketRef.current = io('http://localhost:5000');

    socketRef.current.on('prediction_result', (data) => {
      if (data.word && data.word !== '...') {
        setSubtitle(data.word);

        const porcentaje = (data.confidence).toFixed(1);
        setConfidence(porcentaje);
      }

      setLandmarks(data.landmarks);
      isProcessingRef.current = false;
    });

    socketRef.current.on('word_added', (data) => {
      setWordBuffer(data.buffer);
    });

    socketRef.current.on('translation_result', (data) => {
      setSentence(data.sentence);
      setWordBuffer([]);
      setIsTranslating(false);
      setSubtitle(data.sentence);
      synthesizeAndPlaySpeech(data.sentence, selectedVoiceRef.current);
    });

    return () => {
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  const speakSentence = () => {
    if (socketRef.current) {
      console.log("Solicitando traducción manual al servidor...");
      socketRef.current.emit('trigger_translation');
    }
  };

  // Limpiar buffer del servidor al pulsar Delete
  const handleDelete = () => {
    setSentence('');
    setWordBuffer([]);
    setSubtitle("...");
    if (socketRef.current) {
      socketRef.current.emit('clear_buffer');
    }
  };

  // Streaming de fotogramas
  useEffect(() => {
    let interval;
    if (isAiActive && !isCameraOff) {
      interval = setInterval(() => {
        if (isProcessingRef.current) return;

        if (videoRef.current && canvasRef.current) {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video.videoWidth > 0) {
            isProcessingRef.current = true;

            const MAX_WIDTH = 480;
            const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
            canvas.width = video.videoWidth * scale;
            canvas.height = video.videoHeight * scale;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const base64Frame = canvas.toDataURL('image/jpeg', 0.4);
            if (socketRef.current) {
              socketRef.current.emit('video_frame', base64Frame);
            } else {
              isProcessingRef.current = false;
            }
          }
        }
      }, 50);
    } else if (!isAiActive) {
      // Si apagamos la IA, limpiamos la pantalla sin pisar el estado si el micro está escuchando
      setSubtitle(prev => (isMuted ? "..." : prev));
      setLandmarks([]);
      isProcessingRef.current = false;
    }
    return () => clearInterval(interval);
  }, [isAiActive, isCameraOff, isMuted]);

  // Overlay de puntos clave
  useEffect(() => {
    if (!overlayRef.current || !videoRef.current) return;
    const canvas = overlayRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Dibuja los puntos sobre la imagen reflejada de la webcam
    if (showMesh && landmarks.length > 0) {
      landmarks.forEach(hand => {
        ctx.fillStyle = '#06b6d4'; // Color azul/cian configurado
        hand.landmarks.forEach(lm => {
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    }
  }, [landmarks, showMesh]);

  // Reconocimiento de Voz (Speech-to-Text integrado de forma síncrona)
  useEffect(() => {
    if (!recognition) return;

    if (!isMuted) {
      try {
        recognition.start();
      } catch (e) {
        console.log("El recognition ya estaba activo");
      }
    } else {
      recognition.stop();
    }

    recognition.onresult = (event) => {
      const currentResultIndex = event.resultIndex;
      const transcript = event.results[currentResultIndex][0].transcript;
      setSubtitle(transcript); // Inyecta la voz directamente en el subtítulo común
    };

    recognition.onend = () => {
      if (!isMuted) {
        try {
          recognition.start();
        } catch (e) { }
      }
    };

    recognition.onerror = (event) => {
      console.error("⚠️ Error en SpeechRecognition:", event.error);
    };

    return () => {
      recognition.stop();
    };
  }, [isMuted]);

  return (
    <div className="h-screen bg-zinc-950 text-white flex flex-col font-sans overflow-hidden">
      {/* Header */}
      <header className="p-4 border-b border-white/10 flex justify-between items-center bg-zinc-900/50">
        <h1 className="text-xl font-bold text-cyan-400">Sign-link traductor</h1>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition-all ${showSettings ? 'bg-cyan-600' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Área Principal de Vídeo */}
      <div className="flex-1 min-h-0 p-4 flex bg-black relative justify-center items-center">
        <main className="flex-1 h-full min-h-0 flex items-center justify-center relative">

          {/* Contenedor del Vídeo Feed */}
          <div className={`relative aspect-video max-w-6xl max-h-full w-auto h-auto bg-zinc-900 rounded-2xl overflow-hidden shadow-2xl transition-all duration-300 border-2 ${isAiActive
            ? 'border-cyan-500 shadow-[0_0_30px_rgba(6,182,212,0.3)]'
            : 'border-white/5'
            }`}>

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted // CRITICAL: Silencia el retorno del propio PC para evitar acoples de sonido e interferencias
              className={`w-full h-full object-cover ${isCameraOff ? 'hidden' : 'block'}`}
              style={{ transform: 'scaleX(-1)' }}
            />

            <canvas ref={canvasRef} className="hidden" />

            <canvas
              ref={overlayRef}
              className={`absolute top-0 left-0 w-full h-full object-cover pointer-events-none ${isCameraOff ? 'hidden' : 'block'}`}
            />

            {/* Subtítulos independientes del Modo IA */}
            {showSubtitle && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-11/12 text-center pointer-events-none z-10">
                <p className="text-base sm:text-lg md:text-xl font-bold text-white drop-shadow-[0_2px_4px_rgba(0,0,0,0.9)] tracking-wide">
                  {subtitle.toUpperCase()}
                </p>
              </div>
            )}

            {/* Marcador de Confianza de Predicción */}
            {showConfidence && isAiActive && (
              <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm border border-cyan-500/30 px-2 py-1 rounded-xl flex items-center gap-1.5 animate-in fade-in zoom-in duration-300 pointer-events-none z-10">
                <span className="text-[9px] font-bold text-cyan-400 uppercase tracking-wider">Conf:</span>
                <span className="text-sm font-mono font-bold text-white">{confidence}%</span>
              </div>
            )}
          </div>
        </main>

        {/* Panel Lateral de Ajustes (Settings) */}
        {showSettings && (
          <aside className="w-80 h-full overflow-y-auto bg-zinc-900 border-l border-white/10 p-6 animate-in slide-in-from-right duration-200">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-cyan-400">Settings</h2>
              <button onClick={() => setShowSettings(false)} className="text-zinc-500 hover:text-white">✕</button>
            </div>

            <div className="space-y-4">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Voice Settings</label>

              <div className="flex items-center justify-between bg-zinc-800/30 p-4 rounded-2xl border border-white/5 hover:border-white/10 transition-colors mb-2">
                <div className="flex flex-col">
                  <span className="text-sm font-medium">IA Voice</span>
                  <span className="text-[10px] text-zinc-500 italic">Speak words detected</span>
                </div>
                <button
                  onClick={() => setIsAiVoiceActive(!isAiVoiceActive)}
                  className={`w-12 h-6 rounded-full transition-all relative ${isAiVoiceActive ? 'bg-cyan-500' : 'bg-zinc-700'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${isAiVoiceActive ? 'left-7' : 'left-1'}`} />
                </button>
              </div>

              <label className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mt-2">Voice Type</label>
              <div className="flex flex-col gap-3">
                <button
                  onClick={() => setSelectedVoice('male')}
                  className={`w-full p-4 rounded-xl border transition-all flex items-center justify-between ${selectedVoice === 'male'
                    ? 'bg-cyan-600/10 border-cyan-500 text-cyan-400'
                    : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:bg-zinc-800'
                    }`}
                >
                  <span className="font-medium">Male</span>
                </button>

                <button
                  onClick={() => setSelectedVoice('female')}
                  className={`w-full p-4 rounded-xl border transition-all flex items-center justify-between ${selectedVoice === 'female'
                    ? 'bg-cyan-600/10 border-cyan-500 text-cyan-400'
                    : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:bg-zinc-800'
                    }`}
                >
                  <span className="font-medium">Female</span>
                </button>
              </div>

              <div className="space-y-4 pt-4">
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-widest">IA analysis</label>

                <div className="flex items-center justify-between bg-zinc-800/30 p-4 rounded-2xl border border-white/5 hover:border-white/10 transition-colors">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">IA hand points</span>
                    <span className="text-[10px] text-zinc-500 italic">Visualize hand landmarks</span>
                  </div>
                  <button
                    onClick={() => setShowMesh(!showMesh)}
                    className={`w-12 h-6 rounded-full transition-all relative ${showMesh ? 'bg-cyan-500' : 'bg-zinc-700'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${showMesh ? 'left-7' : 'left-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between bg-zinc-800/30 p-4 rounded-2xl border border-white/5 hover:border-white/10 transition-colors">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">Confidence</span>
                    <span className="text-[10px] text-zinc-500 italic">Show IA confidence on screen</span>
                  </div>
                  <button
                    onClick={() => setShowConfidence(!showConfidence)}
                    className={`w-12 h-6 rounded-full transition-all relative ${showConfidence ? 'bg-cyan-600' : 'bg-zinc-700'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${showConfidence ? 'left-7' : 'left-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between bg-zinc-800/30 p-4 rounded-2xl border border-white/5 hover:border-white/10 transition-colors">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">Subtitles</span>
                    <span className="text-[10px] text-zinc-500 italic">Show subtitles on screen</span>
                  </div>
                  <button
                    onClick={() => setShowSubtitle(!showSubtitle)}
                    className={`w-12 h-6 rounded-full transition-all relative ${showSubtitle ? 'bg-cyan-500' : 'bg-zinc-700'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${showSubtitle ? 'left-7' : 'left-1'}`} />
                  </button>
                </div>
              </div>
            </div>
          </aside>
        )}
      </div>

      {/* Panel Inferior: Muestra de la Frase Acumulada y Acciones */}
      {isAiActive && (
        <div className="px-6 py-3 bg-zinc-900 border-t border-white/10 flex items-center gap-4">
          <div className="flex-1 text-sm min-w-0">
            {isTranslating ? (
              <p className="text-cyan-400 animate-pulse">Traduciendo glosas...</p>
            ) : sentence ? (
              <p className="text-white">{sentence}</p>
            ) : wordBuffer.length > 0 ? (
              <p className="text-zinc-300">
                {wordBuffer.map((w, i) => (
                  <span key={i} className="inline-block bg-zinc-700 text-cyan-300 rounded-md px-2 py-0.5 mr-1 mb-1 font-mono text-xs">
                    {w}
                  </span>
                ))}
              </p>
            ) : (
              <p className="text-zinc-500 italic">Empieza a signar...</p>
            )}
          </div>
          <button
            onClick={handleDelete}
            className="px-4 py-2 rounded-xl bg-zinc-700 hover:bg-zinc-600 text-sm"
          >
            Delete
          </button>
          <button
            onClick={() => { setIsTranslating(true); speakSentence(); }}
            className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-sm font-bold"
          >
            Read sentence
          </button>
        </div>
      )}

      {/* Barra de Controles de Hardware (Footer) */}
      <footer className="p-4 bg-zinc-950 border-t border-white/5 flex justify-center gap-4">
        <button onClick={() => setIsMuted(!isMuted)}
          className={`px-6 py-3 rounded-2xl transition-all active:scale-95 ${isMuted ? 'bg-red-600 hover:bg-red-500' : 'bg-zinc-800 hover:bg-zinc-700'}`}
        >
          {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
        </button>

        <button onClick={() => setIsCameraOff(!isCameraOff)}
          className={`px-6 py-3 rounded-2xl transition-all active:scale-95 ${isCameraOff ? 'bg-red-600 hover:bg-red-500' : 'bg-zinc-800 hover:bg-zinc-700'}`}
        >
          {isCameraOff ? <VideoOff size={24} /> : <Video size={24} />}
        </button>

        <button
          onClick={() => setIsAiActive(!isAiActive)}
          className={`px-8 py-3 rounded-2xl font-bold shadow-lg transition-all active:scale-95 ${isAiActive
            ? 'bg-cyan-50 text-black shadow-cyan-500/40'
            : 'bg-cyan-600 text-white hover:bg-cyan-500 shadow-cyan-500/20'
            }`}
        >
          <Hand size={20} />
        </button>
      </footer>
    </div>
  );
}

export default App;