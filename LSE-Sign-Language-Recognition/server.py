import os
import cv2
import base64
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
from tensorflow.keras.models import load_model

# Flask and Socket.IO Configuration 
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# MediaPipe Configuration
mp_hands = mp.solutions.hands

# Load Model
print("Loading model...")
model = load_model('bilstm_model.h5')

if os.path.exists('label_map_propio.npy'):
    label_map = np.load('label_map_propio.npy', allow_pickle=True).item()
    list_actions = {v: k for k, v in label_map.items()}
    print(f"Words trained: {list(list_actions.values())}")
else:
    print("ERROR: label_map_propio.npy not found.")
    exit()

# State Variables 
threshold = 0.7
MAX_FRAMES = 30
MIN_CONSECUTIVE = 5  # Predictions in a row required to confirm a sign

# Dictionary to store sequence AND MediaPipe instance per client (sid)
client_data = {}

def process_frame(frame_bytes, hands_instance):
    # Decode Base64 image to OpenCV matrix
    np_img = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Mirror the image horizontally
    frame = cv2.flip(frame, 1)
    
    # MediaPipe requires RGB format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_instance.process(img_rgb)
    
    # Initialize 126 zeros array and variables
    frame_data = np.zeros(126)
    landmarks_export = []

    # Process MediaPipe Hands results
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i > 1: break
            
            hand_export = []
            
            label = results.multi_handedness[i].classification[0].label
            start_idx = 0 if label == 'Left' else 63
            
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
    # Create a dedicated MediaPipe instance and sequence list for this client
    client_data[sid] = {
        'sequence': [],
        'hands': mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7),
        'consecutive_count': 0,   # How many times the same word has been predicted in a row
        'last_prediction': None   # Last predicted word (for streak tracking)
    }

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in client_data:
        client_data[sid]['hands'].close()  # Free MediaPipe resources
        del client_data[sid]

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid

    if sid not in client_data:
        return

    # Extract Base64 header
    header, encoded = data.split(",", 1)
    frame_bytes = base64.b64decode(encoded)

    # Use this client's own MediaPipe instance
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

        if word == "(Reposo)" or confidence < threshold:
            # Rest pose or low confidence: reset streak, stay silent
            client_data[sid]['consecutive_count'] = 0
        else:
            if word == client_data[sid]['last_prediction']:
                client_data[sid]['consecutive_count'] += 1
            else:
                # Different word: restart streak
                client_data[sid]['consecutive_count'] = 1
                client_data[sid]['last_prediction'] = word

            if client_data[sid]['consecutive_count'] >= MIN_CONSECUTIVE:
                # Sign confirmed — emit it and reset so the next sign starts fresh
                prediction_result['word'] = word
                prediction_result['confidence'] = int(confidence * 100)
                client_data[sid]['sequence'] = []
                client_data[sid]['consecutive_count'] = 0

    emit('prediction_result', prediction_result)

if __name__ == '__main__':
    print("Starting Socket.IO server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)