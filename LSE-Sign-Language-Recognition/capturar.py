import cv2
import mediapipe as mp
import numpy as np
import os
import time

ACCION_A_GRABAR = "Cual"  # Word to record
DATA_PATH = f"C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition/Database_propio/{ACCION_A_GRABAR}"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Look for the file number to avoid overwriting
existentes = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
# If you already had files, start from the next one
contador = len(existentes) 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f"PREPARADO PARA GRABAR: {ACCION_A_GRABAR.upper()}")
print("Instrucciones:")
print("1. Ponte en posición.")
print("2. Pulsa 'S' para grabar una ráfaga (1 segundo).")
print("3. Pulsa 'Q' para terminar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    cv2.putText(frame, f"Carpeta: {ACCION_A_GRABAR} | Muestras: {contador}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow('Capturador de TFG', frame)

    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('s'):
        print(f"Grabando muestra {contador}...")
        secuencia_temporal = []
        
        # Record hands in 30 frames
        for frame_num in range(30):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            puntos_frame = np.zeros(126)
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    if i > 1: break
                    label = results.multi_handedness[i].classification[0].label
                    idx_inicio = 0 if label == 'Left' else 63
                    for j, lm in enumerate(hand_lms.landmark):
                        idx = idx_inicio + (j * 3)
                        puntos_frame[idx:idx+3] = [lm.x, lm.y, lm.z]
            
            secuencia_temporal.append(puntos_frame)
            
            # Red dot to indicate recording
            cv2.circle(frame, (frame.shape[1]-30, 30), 15, (0, 0, 255), -1)
            cv2.imshow('Capturador de TFG', frame)
            cv2.waitKey(1)

        # Save with a unique name
        file_name = f"{ACCION_A_GRABAR}_new_{contador}.npy"
        np.save(os.path.join(DATA_PATH, file_name), np.array(secuencia_temporal))
        contador += 1
        print("¡Guardado!")

    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()