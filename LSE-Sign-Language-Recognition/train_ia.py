import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Rutas y Parámetros
DATA_PATH = "C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition/Database_propio"
OUTPUT_PATH = "C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition"
MAX_FRAMES = 30 
NUM_FEATURES = 126  # 21 puntos * 3 coords * 2 manos
AUGMENTATION_MULTIPLIER = 4  # Cantidad de clones aumentados por cada secuencia real

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
    
    augmented[:, 0::3] += shift_x  # Coordenadas X
    augmented[:, 1::3] += shift_y  # Coordenadas Y
        
    return augmented

# Carga de datos base
print("Escaneando dataset (extrayendo únicamente archivos base)...")

sequences, labels = [], []
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
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    for file_path in npy_files:
        try:
            seq = np.load(file_path)
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
            continue
            
        if len(seq) == 0:
            continue
            
        # Estandarizar la longitud de la secuencia real
        seq = pad_or_truncate_sequence(seq, MAX_FRAMES)
        
        # Almacenamos únicamente el archivo real original
        sequences.append(seq)
        labels.append(label_id)

X_raw = np.array(sequences)
y_raw = np.array(labels)

# Guardar el mapa de etiquetas mapeado dinámicamente
np.save(os.path.join(OUTPUT_PATH, "label_map_propio.npy"), label_map)

print(f" -> Muestras base originales cargadas: {X_raw.shape[0]}")
print(f" -> Glosas detectadas ({len(label_map)}): {list(label_map.keys())}")

# Extracción del conjunto de validación
# Dividimos primero para aislar un 20% de datos reales puros que la IA jamás verá al entrenar
# El parámetro 'stratify' garantiza un reparto equitativo de muestras por clase
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

print(f" -> Muestras base destinadas a Entrenamiento: {len(X_train_raw)}")
print(f" -> Muestras base destinadas a Validación Pura: {len(X_val_raw)}")

# Aumento de datos sobre el set de entrenamiento sin tocar el de validacion
print("\nAplicando técnicas de Data Augmentation sobre el conjunto de entrenamiento...")
X_train_augmented = []
y_train_augmented = []

for seq, label in zip(X_train_raw, y_train_raw):
    # Guardar el archivo real base en el set de entrenamiento
    X_train_augmented.append(seq)
    y_train_augmented.append(label)
    
    # Generar los clones sintéticos con ruido exclusivamente para el entrenamiento
    for _ in range(AUGMENTATION_MULTIPLIER):
        aug_seq = augment_sequence(seq)
        X_train_augmented.append(aug_seq)
        y_train_augmented.append(label)

# Conversión a arrays finales transformando etiquetas a matrices categóricas (One-Hot)
X_train = np.array(X_train_augmented)
y_train = to_categorical(y_train_augmented, num_classes=len(label_map)).astype(int)

# El conjunto de validación se procesa en limpio, sin alteración de ruido ni clones
X_val = X_val_raw
y_val = to_categorical(y_val_raw, num_classes=len(label_map)).astype(int)

print("Estructura definitiva de datos completada con éxito:")
print(f" -> Matriz X_train (Con aumento): {X_train.shape}, y_train: {y_train.shape}")
print(f" -> Matriz X_val (Reales limpios): {X_val.shape}, y_val: {y_val.shape}\n")

# Arquitectura del modelo BilSTM
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

# Ejecucion del entrenamiento
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

# Guardar el modelo óptimo final
model.save(os.path.join(OUTPUT_PATH, 'bilstm_model.h5'))
print("\nModelo guardado de forma segura como: bilstm_model.h5")

# Guardar logs de entrenamiento para graficar pérdida/precisión en la memoria
np.save(os.path.join(OUTPUT_PATH, 'historial_entrenamiento.npy'), history.history)
print("Historial de entrenamiento guardado como: historial_entrenamiento.npy")