import os
import cv2
import base64
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
from tensorflow.keras.models import load_model

# Importamos el gestor de traducción
import sign_buffer_manager

# Configuración de flask y socketio
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuración de mediapipe
mp_hands = mp.solutions.hands

# Cargar el modelo
print("Loading model...")
model = load_model('bilstm_model.h5')

if os.path.exists('label_map_propio.npy'):
    label_map = np.load('label_map_propio.npy', allow_pickle=True).item()
    list_actions = {v: k for k, v in label_map.items()} # Dar la vuelta al diccionario (HOLA: 0 -> 0: HOLA)
    print(f"Words trained: {list(list_actions.values())}")
else:
    print("ERROR: label_map_propio.npy not found.")
    exit()

threshold = 0.7
MAX_FRAMES = 30
MIN_CONSECUTIVE = 5  # Predicciones consecutivas requeridas para confirmar una seña

# Soporte multiusuario con diccionario
client_data = {}

def process_frame(frame_bytes, hands_instance):
    # Decodificar la imagen Base64 a una matriz OpenCV
    np_img = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Efecto espejo para comodidad
    frame = cv2.flip(frame, 1)
    
    # MediaPipe requiere formato RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_instance.process(img_rgb)
    
    # Inicializar array de 126 ceros y variables
    frame_data = np.zeros(126)
    landmarks_export = []

    # Procesar los resultados de MediaPipe Hands
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i > 1: break # Reconocer maximo 2 manos
            
            hand_export = []
            
            label = results.multi_handedness[i].classification[0].label
            start_idx = 0 if label == 'Left' else 63 # Separar izquierda de derecha)
            
            for j, lm in enumerate(hand_lms.landmark):
                idx = start_idx + (j * 3)
                frame_data[idx:idx+3] = [lm.x, lm.y, lm.z]
                hand_export.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                
            landmarks_export.append({
                'label': label,
                'landmarks': hand_export
            })

    return frame_data, landmarks_export

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    # Crear una instancia dedicada de MediaPipe y una lista de secuencias para este cliente
    client_data[sid] = {
        'sequence': [],
        'hands': mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7),
        'consecutive_count': 0,   # Cuántas veces seguidas se ha predicho la misma palabra
        'last_prediction': None   # Última palabra detectada (evitar duplicados)
    }

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in client_data:
        client_data[sid]['hands'].close()  # Liberar recursos de MediaPipe
        del client_data[sid]

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid

    if sid not in client_data:
        return

    # Extraer la cabecera Base64
    header, encoded = data.split(",", 1)
    frame_bytes = base64.b64decode(encoded)

    # Usar la instancia de MediaPipe propia de este cliente
    frame_data, landmarks_export = process_frame(frame_bytes, client_data[sid]['hands'])

    client_data[sid]['sequence'].append(frame_data)
    client_data[sid]['sequence'] = client_data[sid]['sequence'][-MAX_FRAMES:]

    prediction_result = {
        'word': "...",
        'confidence': 0,
        'landmarks': landmarks_export
    }

    if len(client_data[sid]['sequence']) == MAX_FRAMES:
        sequence_np = np.expand_dims(client_data[sid]['sequence'], axis=0)
        res = model.predict(sequence_np, verbose=0)[0]
        max_idx = np.argmax(res)
        confidence = float(res[max_idx])
        word = list_actions[max_idx]

        # Filtro de reposo estricto con los paréntesis de tu dataset entrenado
        if word == "(Reposo)" or confidence < threshold:
            client_data[sid]['consecutive_count'] = 0
        else:
            if word == client_data[sid]['last_prediction']:
                client_data[sid]['consecutive_count'] += 1
            else:
                client_data[sid]['consecutive_count'] = 1
                client_data[sid]['last_prediction'] = word

            if client_data[sid]['consecutive_count'] >= MIN_CONSECUTIVE:
                prediction_result['word'] = word
                prediction_result['confidence'] = int(confidence * 100)
                
                # Se guarda la palabra de forma segura en la lista interna
                sign_buffer_manager.add_word(word)
                emit('word_added', {'buffer': sign_buffer_manager.buffer})
                
                
                client_data[sid]['sequence'] = []
                client_data[sid]['consecutive_count'] = 0

    emit('prediction_result', prediction_result)

# Evento disparador manual de traducción LLM
@socketio.on('trigger_translation')
def handle_trigger_translation():
    sid = request.sid
    print(f"[Socket.IO] El cliente {sid} ha solicitado la traducción de la frase.")
    
    frase_final = sign_buffer_manager.translate_current_buffer()
    
    if frase_final:
        print(f"[Socket.IO] Enviando frase estructurada a la interfaz web: '{frase_final}'")
        # Se le inyecta la frase en el canal global para actualizar la UI de React
        emit('translation_result', {'sentence': frase_final})

@socketio.on('clear_buffer')
def handle_clear_buffer():
    sign_buffer_manager.buffer.clear()
    sign_buffer_manager.last_word = None
    print(f"[Socket.IO] Buffer limpiado por el cliente {request.sid}")

if __name__ == '__main__':
    print("Starting Socket.IO server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)