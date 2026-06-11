import cv2
import mediapipe as mp
import csv
import os

# Mediapipe
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    datalist = []

    print (f"Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z]) # Guardar X, Y, Z para cada uno de los 21 puntos
                datalist.append(landmarks)
        
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(datalist)
    
    cap.release()
    print(f"Finished processing. Data saved to: {output_csv}")
