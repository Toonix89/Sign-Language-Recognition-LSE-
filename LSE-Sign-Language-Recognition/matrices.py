import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Rutas a tu proyecto
OUTPUT_PATH = "C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition"
DATA_PATH   = os.path.join(OUTPUT_PATH, "Database_propio")   # carpeta con subcarpetas por glosa
MODEL_PATH  = os.path.join(OUTPUT_PATH, "bilstm_model.h5")
GRAFICOS    = os.path.join(OUTPUT_PATH, "graficos")

# Tiene que coincidir con el orden con el que se entrenó el modelo.
# Si guardaste un label_encoder o una lista de clases, cárgala desde ahí.
ACCIONES = [
    "(Reposo)", "Adios", "Bien", "Cual", "Gracias",
    "Hola", "No", "Nombre", "Por favor", "Si",
    "Tu", "Yo", "¿Qué tal"
]

FRAMES_POR_SECUENCIA = 30
RANDOM_SEED = 42

os.makedirs(GRAFICOS, exist_ok=True)

# ============================================================
# 1. CARGAR DATASET (igual que en tu train_ia.py)
# ============================================================
print("Cargando dataset...")
secuencias, etiquetas = [], []

for idx, accion in enumerate(ACCIONES):
    carpeta = os.path.join(DATA_PATH, accion)
    if not os.path.isdir(carpeta):
        print(f"  AVISO: No existe la carpeta '{carpeta}', se omite.")
        continue
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(".npy")])
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        secuencia = np.load(ruta)
        if secuencia.shape == (FRAMES_POR_SECUENCIA, 126):
            secuencias.append(secuencia)
            etiquetas.append(idx)

X = np.array(secuencias)                          # (N, 30, 126)
y = np.array(etiquetas)                           # (N,)

print(f"  Total muestras cargadas: {len(X)}")

# ============================================================
# 2. REPRODUCIR EL MISMO SPLIT QUE EN EL ENTRENAMIENTO
# ============================================================
_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=y          # importante para que el split sea proporcional
)

print(f"  Muestras de validación: {len(X_val)}")

# ============================================================
# 3. CARGAR MODELO Y PREDECIR
# ============================================================
print("Cargando modelo...")
modelo = load_model(MODEL_PATH)

y_pred_proba = modelo.predict(X_val, verbose=0)   # (N_val, 13)
y_pred       = np.argmax(y_pred_proba, axis=1)    # clase predicha
y_true       = y_val

# ============================================================
# 4. INFORME POR CLASE (precision, recall, F1)
# ============================================================
print("\n--- Informe de clasificación ---")
report = classification_report(
    y_true, y_pred,
    target_names=ACCIONES,
    digits=3
)
print(report)

# Guardarlo también como .txt para tenerlo accesible
with open(os.path.join(GRAFICOS, "informe_clasificacion.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("  Informe guardado en graficos/informe_clasificacion.txt")

# ============================================================
# 5. MATRIZ DE CONFUSIÓN - Valores absolutos
# ============================================================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ACCIONES,
    yticklabels=ACCIONES,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Matriz de confusión — valores absolutos", fontsize=14, pad=12)
ax.set_xlabel("Predicción", fontsize=11)
ax.set_ylabel("Etiqueta real", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "matriz_confusion.png"), dpi=150)
plt.show()
print("  Guardada: graficos/matriz_confusion.png")

# ============================================================
# 6. MATRIZ DE CONFUSIÓN - Normalizada (% por fila = recall)
# ============================================================
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=ACCIONES,
    yticklabels=ACCIONES,
    linewidths=0.5,
    vmin=0, vmax=1,
    ax=ax
)
ax.set_title("Matriz de confusión — normalizada por fila (recall)", fontsize=14, pad=12)
ax.set_xlabel("Predicción", fontsize=11)
ax.set_ylabel("Etiqueta real", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "matriz_confusion_normalizada.png"), dpi=150)
plt.show()
print("  Guardada: graficos/matriz_confusion_normalizada.png")

print("\nListo. Todos los gráficos generados.")