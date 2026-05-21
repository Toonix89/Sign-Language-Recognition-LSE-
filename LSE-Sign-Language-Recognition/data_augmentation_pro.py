import numpy as np
import os
import random

# --- CONFIGURACIÓN ---
DATA_PATH = "C:/TFG/Database"
actions = ['Adios']
num_variantes = 4  # Por cada video original, crearemos 4 nuevos (total 5 por video)

def augment_data(data):
    # 1. Ruido Gaussiano (Noise)
    noise = np.random.normal(0, 0.007, data.shape)
    variant = data + noise
    
    # 2. Desplazamiento aleatorio (Shift)
    # Movemos ligeramente las coordenadas X e Y
    shift_x = np.random.uniform(-0.03, 0.03)
    shift_y = np.random.uniform(-0.03, 0.03)
    
    # Aplicamos el desplazamiento a todas las coordenadas X (índices 0, 3, 6...)
    variant[:, 0::3] += shift_x
    # Aplicamos el desplazamiento a todas las coordenadas Y (índices 1, 4, 7...)
    variant[:, 1::3] += shift_y
    
    return variant

print("Iniciando Data Augmentation...")

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    # Listamos solo los archivos originales (evitamos procesar los aumentados si ya existen)
    files = [f for f in os.listdir(action_folder) if f.endswith('.npy') and 'aug' not in f]
    
    print(f"Procesando {action}... ({len(files)} archivos originales)")
    
    for file_name in files:
        full_path = os.path.join(action_folder, file_name)
        res = np.load(full_path)
        
        for i in range(num_variantes):
            augmented_res = augment_data(res)
            # Guardamos con un nombre nuevo
            new_name = f"aug_{i}_{file_name}"
            np.save(os.path.join(action_folder, new_name), augmented_res)

print("¡Proceso finalizado! Tu dataset es ahora 5 veces más grande.")