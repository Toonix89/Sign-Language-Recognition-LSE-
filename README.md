# 🤟 LSE-Sign-Language-Recognition
**Traductor de Lengua de Signos Española (LSE) en tiempo real mediante Deep Learning.**

Este proyecto implementa un sistema de visión artificial capaz de traducir gestos de la LSE a texto de manera instantánea. Se utilizan extracción de puntos para analizar los gestos de la mano y el uso de redes neuronales para clasificar etiquetas
---

## 🚀 Características
- **Detección Multi-Mano:** Seguimiento de 21 puntos clave por mano mediante MediaPipe.
- **Análisis Temporal:** Clasificación de secuencias de movimiento mediante redes **LSTM (Long Short-Term Memory)**.

---

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.11
- **Deep Learning:** TensorFlow 2.15 / Keras
- **Visión Artificial:** MediaPipe Holistic (Heavy Model)
- **Procesamiento de Datos:** NumPy, Pandas, Scikit-learn
- **Interfaz:** OpenCV

---

## 📊 Metodología y Modelo
El sistema procesa secuencias de **15/30 frames** para identificar el signo. Cada frame es convertido en un vector de **126 características** (coordenadas X, Y, Z de ambas manos).

## 📦 Requisitos y Versiones del Sistema
Para asegurar la compatibilidad de los modelos `.h5` y el funcionamiento de MediaPipe, se deben utilizar las siguientes versiones exactas:

* **Python:** `3.11.9` (Recomendado para evitar conflictos con TensorFlow 2.15)
* **TensorFlow:** `2.15.0` 
* **MediaPipe:** `0.10.9` 
* **OpenCV-Python:** `4.9.0.80`
* **NumPy:** `1.26.4` (Versión compatible con TensorFlow 2.15 sin errores de tipos)
* **Protobuf:** `3.20.3` (Para evitar errores de serialización en MediaPipe)