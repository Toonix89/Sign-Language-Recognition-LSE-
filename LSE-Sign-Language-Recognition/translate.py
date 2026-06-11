# --- Traductor LSE - Reconocimiento de Lengua de Señas en Tiempo Real ---

import os
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Configuración del modelo y dibujo de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Cargar el modelo BiLSTM y el mapa de etiquetas
print("Loading model...")
model = load_model('bilstm_model.h5')

if os.path.exists('label_map_propio.npy'):
    label_map = np.load('label_map_propio.npy', allow_pickle=True).item()
    # Invertir el diccionario a id -> palabra
    list_actions = {v: k for k, v in label_map.items()}
    print(f"Words trained: {list(list_actions.values())}")
else:
    print("ERROR: label_map_propio.npy not found. Please train the model first.")
    exit()

# Secuencia y umbral
sequence = []
sentence = ""
confidence = 0
threshold = 0.7
MAX_FRAMES = 30
MIN_CONSECUTIVE = 5   # Predicciones consecutivas requeridas para confirmar una seña
consecutive_count = 0 # Contador de racha actual
last_prediction = None  # Última palabra predicha (para seguimiento de racha)

# Iniciar cámara
cap = cv2.VideoCapture(0)
print("Translator started. Press 'q' to exit.")

# Inicializar el modelo de Manos
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe Hands
        results = hands.process(img_rgb)
        
        # Array de 126 valores (2 manos * 21 puntos * 3 coordenadas)
        frame_data = np.zeros(126) 
        
        # --- DIBUJO Y EXTRACCIÓN ---
        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                # Dibujar puntos de referencia (landmarks) en el fotograma
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if i > 1: break # Máximo 2 manos
                
                label = results.multi_handedness[i].classification[0].label
                start_idx = 0 if label == 'Left' else 63
                
                for j, lm in enumerate(hand_lms.landmark):
                    idx = start_idx + (j * 3)
                    frame_data[idx:idx+3] = [lm.x, lm.y, lm.z]
        else:
            # SIN MANOS: Enviar ceros
            frame_data = np.zeros(126)

        # Añadir a la secuencia
        sequence.append(frame_data)
        sequence = sequence[-MAX_FRAMES:]

        # --- PREDICCIÓN ---
        if len(sequence) == MAX_FRAMES:
            # expand_dims para tener la forma (1, 30, 126)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            max_idx = np.argmax(res)
            confidence = res[max_idx]
            word = list_actions[max_idx]

            if word == "(Reposo)" or confidence < threshold:
                # Pose de reposo o baja confianza: reiniciar racha, permanecer en silencio
                consecutive_count = 0
                sentence = "..."
            else:
                if word == last_prediction:
                    consecutive_count += 1
                else:
                    # Palabra diferente: reiniciar racha
                    consecutive_count = 1
                    last_prediction = word

                if consecutive_count >= MIN_CONSECUTIVE:
                    # Seña confirmada — mostrarla y reiniciar para que la siguiente comience de cero
                    sentence = word
                    sequence = []
                    consecutive_count = 0
                else:
                    sentence = "..."

        # --- INTERFAZ ---
        # Rectángulo superior para el texto
        cv2.rectangle(frame, (0,0), (640, 45), (245, 117, 16), -1)
        
        # Texto de predicción y porcentaje
        display_text = f"{sentence.upper()} ({int(confidence*100)}%)"
        cv2.putText(frame, display_text, (10, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('TFG - Traductor LSE', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()