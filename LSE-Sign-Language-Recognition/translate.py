# --- Translator LSE - Real Time Sign Language Recognition ---

import os
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# MediaPipe drawing and model configuration
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Load the BiLSTM model and label map
print("Loading model...")
model = load_model('bilstm_model.h5')

if os.path.exists('label_map_propio.npy'):
    label_map = np.load('label_map_propio.npy', allow_pickle=True).item()
    # Invert dictionary to id -> word
    list_actions = {v: k for k, v in label_map.items()}
    print(f"Words trained: {list(list_actions.values())}")
else:
    print("ERROR: label_map_propio.npy not found. Please train the model first.")
    exit()

# Sequence and threshold
sequence = []
sentence = ""
confidence = 0
threshold = 0.7
MAX_FRAMES = 30
MIN_CONSECUTIVE = 5   # Predictions in a row required to confirm a sign
consecutive_count = 0 # Current streak counter
last_prediction = None  # Last predicted word (for streak tracking)

# Start camera
cap = cv2.VideoCapture(0)
print("Translator started. Press 'q' to exit.")

# Initialize Hands model
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = hands.process(img_rgb)
        
        # 126 values array (2 hands * 21 points * 3 coordinates)
        frame_data = np.zeros(126) 
        
        # --- DRAWING AND EXTRACTION ---
        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if i > 1: break # Max 2 hands
                
                label = results.multi_handedness[i].classification[0].label
                start_idx = 0 if label == 'Left' else 63
                
                for j, lm in enumerate(hand_lms.landmark):
                    idx = start_idx + (j * 3)
                    frame_data[idx:idx+3] = [lm.x, lm.y, lm.z]
        else:
            # NO HANDS: Send zeros
            frame_data = np.zeros(126)

        # Add to sequence
        sequence.append(frame_data)
        sequence = sequence[-MAX_FRAMES:]

        # --- PREDICTION ---
        if len(sequence) == MAX_FRAMES:
            # expand_dims to have shape (1, 30, 126)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            max_idx = np.argmax(res)
            confidence = res[max_idx]
            word = list_actions[max_idx]

            if word == "(Reposo)" or confidence < threshold:
                # Rest pose or low confidence: reset streak, stay silent
                consecutive_count = 0
                sentence = "..."
            else:
                if word == last_prediction:
                    consecutive_count += 1
                else:
                    # Different word: restart streak
                    consecutive_count = 1
                    last_prediction = word

                if consecutive_count >= MIN_CONSECUTIVE:
                    # Sign confirmed — show it and reset so the next sign starts fresh
                    sentence = word
                    sequence = []
                    consecutive_count = 0
                else:
                    sentence = "..."

        # --- INTERFACE ---
        # Top rectangle for text
        cv2.rectangle(frame, (0,0), (640, 45), (245, 117, 16), -1)
        
        # Prediction and Percentage text
        display_text = f"{sentence.upper()} ({int(confidence*100)}%)"
        cv2.putText(frame, display_text, (10, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('TFG - Traductor LSE', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()