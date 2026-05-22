import React, { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Video, VideoOff, Settings, Hand, PhoneOff } from 'lucide-react';
import { io } from 'socket.io-client';
function App() {
  const videoRef = useRef(null);                              // Reference to the video element
  const [showSettings, setShowSettings] = useState(false);    // State of settings panel
  const [isMuted, setIsMuted] = useState(false);              // State of microphone 
  const [isCameraOff, setIsCameraOff] = useState(false);      // State of camera 
  const [isAiActive, setIsAiActive] = useState(false);        // State of AI 
  const [isAiVoiceActive, setIsAiVoiceActive] = useState(false); // State of AI Voice
  const [selectedVoice, setSelectedVoice] = useState('male'); // State of selected voice
  const [showMesh, setShowMesh] = useState(true);             // Mediapipe points
  const [showConfidence, setShowConfidence] = useState(true); // % Precision
  const [showSubtitle, setShowSubtitle] = useState(true);     // Subtitles
  const [confidence, setConfidence] = useState(0);           // Precision value
  const [subtitle, setSubtitle] = useState("Esperando IA..."); // Subtitle value
  const [landmarks, setLandmarks] = useState([]);              // Hand landmarks
  const [sentence, setSentence] = useState('');               // Sentence value

  const canvasRef = useRef(null);                             // Hidden canvas for extracting frames
  const overlayRef = useRef(null);                            // Canvas for drawing mesh
  const socketRef = useRef(null);                             // Socket reference
  const isProcessingRef = useRef(false);                      // Control de bloqueo para no saturar al servidor

  // Referencias para evitar clausuras obsoletas en los eventos de Socket.IO
  const isAiVoiceActiveRef = useRef(isAiVoiceActive);
  const selectedVoiceRef = useRef(selectedVoice);

  useEffect(() => {
    isAiVoiceActiveRef.current = isAiVoiceActive;
  }, [isAiVoiceActive]);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    startVideo();

  }, []); // [] Start the video once when components loads

  // Socket.IO Connection
  useEffect(() => {
    socketRef.current = io('http://localhost:5000');

    socketRef.current.on('prediction_result', (data) => {
      setSubtitle(data.word);
      setConfidence(data.confidence);
      setLandmarks(data.landmarks);
      isProcessingRef.current = false; // Liberamos el bloqueo al recibir respuesta
    });

    // Escuchamos la traducción completa de Gemini
    socketRef.current.on('translation_result', (data) => {
      setSentence(data.sentence);
      
      // Si la voz de la IA está activa, lee la frase traducida de forma automática
      if (isAiVoiceActiveRef.current) {
        const voiceName = selectedVoiceRef.current === 'female'
          ? 'es-ES-Neural2-A'
          : 'es-ES-Neural2-B';

        fetch(
          `https://texttospeech.googleapis.com/v1/text:synthesize?key=AIzaSyBeaMCV7iPxCScoTYAw7jVrtE9cu3s8XxA`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              input: { text: data.sentence },
              voice: { languageCode: 'es-ES', name: voiceName },
              audioConfig: { audioEncoding: 'MP3' }
            })
          }
        )
        .then(res => res.json())
        .then(resData => {
          if (resData.audioContent) {
            const audio = new Audio(`data:audio/mp3;base64,${resData.audioContent}`);
            audio.play();
          }
        })
        .catch(err => console.error("Error autoplacing TTS:", err));
      }
    });

    return () => {
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  // Frame sending loop
  useEffect(() => {
    let interval;
    if (isAiActive && !isCameraOff) {
      interval = setInterval(() => {
        // If server is processing previous frame, skip this one
        if (isProcessingRef.current) return;

        if (videoRef.current && canvasRef.current) {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video.videoWidth > 0) {
            isProcessingRef.current = true; // Block until server responds

            // Reduce video resolution drastically to speed up sending and processing
            const MAX_WIDTH = 480;
            const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
            canvas.width = video.videoWidth * scale;
            canvas.height = video.videoHeight * scale;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Use strong JPEG compression (0.4) to lighten Base64
            const base64Frame = canvas.toDataURL('image/jpeg', 0.4);
            if (socketRef.current) {
              socketRef.current.emit('video_frame', base64Frame);
            } else {
              isProcessingRef.current = false;
            }
          }
        }
      }, 50); // Evaluate every 50ms, but only send if free
    } else if (!isAiActive) {
      setSubtitle("IA desactivada");
      setLandmarks([]);
      isProcessingRef.current = false;
    }
    return () => clearInterval(interval);
  }, [isAiActive, isCameraOff]);

  // Mesh drawing loop
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

    if (showMesh && landmarks.length > 0) {
      landmarks.forEach(hand => {
        ctx.fillStyle = '#06b6d4'; // cyan-500
        hand.landmarks.forEach(lm => {
          ctx.beginPath();
          // lm.x and lm.y are normalized [0, 1]
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    }
  }, [landmarks, showMesh]);

  // IA Text-to-Speech Voice
  useEffect(() => {
    // La frase acumulada ahora se maneja directamente por el servidor y Gemini,
    // por lo que no es necesario acumular glosas de forma cruda en el frontend.
  }, [subtitle]);

  // Función para llamar a Google TTS
  const speakSentence = async () => {
    if (!sentence.trim()) return;

    const voiceName = selectedVoice === 'female'
      ? 'es-ES-Neural2-A'
      : 'es-ES-Neural2-B';

    const response = await fetch(
      `https://texttospeech.googleapis.com/v1/text:synthesize?key=AIzaSyBeaMCV7iPxCScoTYAw7jVrtE9cu3s8XxA`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: { text: sentence },
          voice: { languageCode: 'es-ES', name: voiceName },
          audioConfig: { audioEncoding: 'MP3' }
        })
      }
    );

    const data = await response.json();
    const audio = new Audio(`data:audio/mp3;base64,${data.audioContent}`);
    audio.play();
  };

  return (
    <div className="h-screen bg-zinc-950 text-white flex flex-col font-sans overflow-hidden pb-8">
      {/* Header */}
      <header className="p-4 border-b border-white/10 flex justify-between items-center bg-zinc-900/50">
        <h1 className="text-xl font-bold text-cyan-400">Sign-link traductor</h1>
        <div className="flex items-center gap-4">
          {/* Settings Button */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition-all ${showSettings ? 'bg-cyan-600' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main Video Area */}
      <div className="flex-1 p-6 flex items-center justify-center bg-black relative">
        <main className="flex-1 px-6 pt-6 pb-2 flex items-center justify-center bg-black relative">
          <div className={`relative w-full max-w-4xl aspect-video bg-zinc-900 rounded-2xl overflow-hidden shadow-2xl transition-all duration-500 border-2 ${isAiActive
            ? 'border-cyan-500 shadow-[0_0_30px_rgba(6,182,212,0.3)]'
            : 'border-white/5'
            }`}>

            {/* The video tag where your face will be displayed */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className={`w-full h-full object-cover ${isCameraOff ? 'hidden' : 'block'}`}
              style={{ transform: 'scaleX(-1)' }} // Mirrored
            />

            {/* Hidden canvas to extract frames */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Overlay canvas to draw hand landmarks */}
            <canvas
              ref={overlayRef}
              className={`absolute top-0 left-0 w-full h-full object-cover pointer-events-none ${isCameraOff ? 'hidden' : 'block'}`}
            // No longer mirrored here because the coordinates from server.py are already mirrored
            />

            {/* Overlay subtitle */}
            {showSubtitle && (
              <div className="absolute bottom-10 left-1/2 -translate-x-1/2 w-2/3 text-center">
                <p className="text-xl font-medium text-white">{isAiActive ? subtitle.toUpperCase() : "..."}</p>
              </div>
            )}

            {/* Precision confidence indicator */}
            {showConfidence && isAiActive && (
              <div className="absolute top-6 right-6 bg-black/60 backdrop-blur-md border border-cyan-500/30 px-4 py-2 rounded-2xl flex flex-col items-end animate-in fade-in zoom-in duration-300">
                <span className="text-[10px] font-bold text-cyan-500 uppercase tracking-tighter">AI Confidence</span>
                <span className="text-2xl font-mono font-bold text-white">{confidence}%</span>
              </div>
            )}
          </div>
        </main>

        {/* Settings Panel */}
        {showSettings && (
          <aside className="w-80 bg-zinc-900 border-l border-white/10 p-6">
            {/* Settings content */}
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-cyan-400">Settings</h2>
              <button onClick={() => setShowSettings(false)} className="text-zinc-500 hover:text-white">✕</button>
            </div>

            <div className="space-y-4">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Voice Settings</label>

              {/* Switch: IA Voice */}
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
                {/* Male voice option */}
                <button
                  onClick={() => setSelectedVoice('male')}
                  className={`w-full p-4 rounded-xl border transition-all flex items-center justify-between ${selectedVoice === 'male'
                    ? 'bg-cyan-600/10 border-cyan-500 text-cyan-400'
                    : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:bg-zinc-800'
                    }`}
                >
                  <span className="font-medium">Male</span>
                </button>

                {/* Female voice option */}
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

              {/* IA Analysis */}
              <div className="space-y-4 pt-4">
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-widest">IA analysis</label>

                {/* Switch: Hand Landmarks Mediapipe*/}
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

                {/* Switch: IA prediction confidence */}
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

                {/* Switch: Subtitles*/}
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
          </aside>)}
      </div>

      {/* Accumulated sentence and buttons */}
      {isAiActive && (
        <div className="px-6 py-3 bg-zinc-900 border-t border-white/10 flex items-center gap-4">
          <p className="flex-1 text-white text-sm">
            {sentence || 'Empieza a signar...'}
          </p>
          <button
            onClick={() => setSentence('')}
            className="px-4 py-2 rounded-xl bg-zinc-700 hover:bg-zinc-600 text-sm"
          >
            Delete
          </button>
          <button
            onClick={speakSentence}
            className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-sm font-bold"
          >
            Read sentence
          </button>
        </div>
      )}


      {/* Bottom Control Bar */}
      <footer className="p-2 pb-12 bg-black flex justify-center gap-4">

        {/* Microphone button */}
        <button onClick={() => setIsMuted(!isMuted)}
          className={`px-6 py-3 rounded-2xl transition-all active:scale-95 ${isMuted ? 'bg-red-600 hover:bg-red-500' : 'bg-zinc-800 hover:bg-zinc-700'}`}
        >
          {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
        </button>

        {/* Camera video button*/}
        <button onClick={() => setIsCameraOff(!isCameraOff)}
          className={`px-6 py-3 rounded-2xl transition-all active:scale-95 ${isCameraOff ? 'bg-red-600 hover:bg-red-500' : 'bg-zinc-800 hover:bg-zinc-700'}`}
        >
          {isCameraOff ? <VideoOff size={24} /> : <Video size={24} />}
        </button>

        {/* IA mode button*/}
        <button
          onClick={() => setIsAiActive(!isAiActive)}
          className={`px-8 py-3 rounded-2xl font-bold shadow-lg transition-all active:scale-95 ${isAiActive
            ? 'bg-cyan-500 text-black shadow-cyan-500/40'
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