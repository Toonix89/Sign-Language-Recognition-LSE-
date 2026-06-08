import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = "C:/TFG/Sign-Language-Recognition-LSE-/LSE-Sign-Language-Recognition"
PATH_HISTORIAL = os.path.join(OUTPUT_PATH, 'historial_entrenamiento.npy')

# Comprobamos si el archivo existe
if not os.path.exists(PATH_HISTORIAL):
    print(f"ERROR: No se encuentra el archivo {PATH_HISTORIAL}.")
    print("Debes ejecutar el entrenamiento completo al menos una vez para generarlo.")
    exit()

# Cargar el diccionario de métricas
history_dict = np.load(PATH_HISTORIAL, allow_pickle=True).item()

print("Historial cargado con éxito. Generando gráficos...")

# Dibujar las gráficas
plt.figure(figsize=(12, 4))

# Gráfico de Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Train')
plt.plot(history_dict['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico de Loss
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Train')
plt.plot(history_dict['val_loss'], label='Validación')
plt.title('Función de pérdida')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Asegurar que existe la carpeta para guardar la imagen
os.makedirs('graficos', exist_ok=True)
plt.savefig('graficos/training_history.png', dpi=150)
print("Imagen guardada en 'graficos/training_history.png'")

# Truco extra: Muestra la gráfica en una ventana emergente interactiva para hacer zoom
plt.show()