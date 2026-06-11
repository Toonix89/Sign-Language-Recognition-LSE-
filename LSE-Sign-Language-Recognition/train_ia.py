import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



DATA_PATH = f"C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition/Database_propio"
MAX_FRAMES = 30 
NUM_FEATURES = 126 # 21 puntos * 3 coords * 2 manos
AUGMENTATION_MULTIPLIER = 4 # Por cada archivo real, generaremos "x" extras

# 1. Función para rellenar o truncar secuencias
def pad_or_truncate_sequence(seq, max_frames=30):
    seq_len = len(seq)
    if seq_len >= max_frames:
        return seq[:max_frames]
    else:
        padding = np.zeros((max_frames - seq_len, NUM_FEATURES))
        return np.vstack((seq, padding))

# 2. Función de aumento de datos
def augment_sequence(seq):
    augmented = np.copy(seq)
    # Ruido Gaussiano
    noise = np.random.normal(0, 0.007, augmented.shape)
    augmented += noise
    
    # Desplazamiento aleatorio
    shift_x = np.random.uniform(-0.03, 0.03)
    shift_y = np.random.uniform(-0.03, 0.03)
    
    augmented[:, 0::3] += shift_x  # x
    augmented[:, 1::3] += shift_y  # y
        
    return augmented

print("Escaneando dataset...")

sequences, labels = [], []

# Mapeo dinámico de carpetas
label_map = {}
current_label_id = 0

# Iterar sobre las carpetas de palabras en Database_propio
for word_folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, word_folder)
    if not os.path.isdir(folder_path):
        continue
        
    if word_folder not in label_map:
        label_map[word_folder] = current_label_id
        current_label_id += 1
        
    label_id = label_map[word_folder]
    
    # Cargar todos los archivos .npy en la carpeta
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    for file_path in npy_files:
        try:
            seq = np.load(file_path)
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
            continue
            
        if len(seq) == 0:
            continue
            
        # Estandarizar la longitud
        seq = pad_or_truncate_sequence(seq, MAX_FRAMES)
        
        # Original
        sequences.append(seq)
        labels.append(label_id)
        
        # Evitamos el aumento infinito si ya es un archivo aumentado ('aug' en el nombre)
        # Solo aumentamos los archivos "originales" y los "nuevos" que grabaste para darle un pequeño impulso
        if 'aug' not in os.path.basename(file_path):
            for _ in range(AUGMENTATION_MULTIPLIER):
                aug_seq = augment_sequence(seq)
                sequences.append(aug_seq)
                labels.append(label_id)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Guardar el mapa de etiquetas para usar en translate.py
OUTPUT_PATH = "C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition"
np.save(os.path.join(OUTPUT_PATH, "label_map_propio.npy"), label_map)

print(f"Datos cargados. X shape: {X.shape}, y shape: {y.shape}")
print(f"Palabras detectadas ({len(label_map)}): {list(label_map.keys())}")

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

print("Construyendo el modelo BiLSTM...")

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), 
                  input_shape=(MAX_FRAMES, NUM_FEATURES)),
    Dropout(0.3),
    
    Bidirectional(LSTM(256, return_sequences=False, activation='tanh')),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- ENTRENAMIENTO ---
print("Empezando el entrenamiento...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint(os.path.join(OUTPUT_PATH, 'bilstm_model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=32,
    callbacks=callbacks
)

model.save(os.path.join(OUTPUT_PATH, 'bilstm_model.h5'))
print("Modelo guardado como bilstm_model.h5")

np.save(os.path.join(OUTPUT_PATH, 'historial_entrenamiento.npy'), history.history)
print("Historial de entrenamiento guardado como historial_entrenamiento.npy")
