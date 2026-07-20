import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preparar_datos(ruta_dataset):
    """
    Función para cargar, limpiar y preparar el dataset para la red neuronal.
    """
    print(f"Cargando dataset desde: {ruta_dataset}")
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_dataset}. Por favor, descárgalo de Kaggle y colócalo aquí.")
        
    df = pd.read_csv(ruta_dataset)
    
    # 1. Exploración inicial
    print("\n--- Información inicial del dataset ---")
    print(f"Dimensiones: {df.shape}")
    print("\nDistribución de la variable objetivo (Target):")
    print(df['Target'].value_counts())
    
    # 2. Limpieza de datos
    # Revisar si hay valores nulos
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"\nSe encontraron {nulos} valores nulos. Procediendo a limpiarlos...")
        df = df.dropna() # Estrategia simple: eliminar nulos. 
    else:
        print("\nNo se encontraron valores nulos en el dataset.")
        
    # Eliminar columnas irrelevantes si existieran (ej. IDs). En este dataset suele haber una columna 'Unnamed: 0' o similar al inicio
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 3. Separación de características (X) y variable objetivo (y)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # 4. Codificación de la variable objetivo (Target)
    # Target actual: 'Dropout', 'Enrolled', 'Graduate'
    # Lo convertiremos a números: 0, 1, 2
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("\nMapeo de clases de la variable objetivo:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    # 5. División del conjunto de datos (Entrenamiento y Prueba)
    # 80% para entrenar, 20% para probar
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

    # 6. Normalización (Escalado de características)
    # Las Redes Neuronales funcionan mucho mejor cuando los datos están normalizados (media 0, varianza 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Ojo: se transforma usando las métricas del conjunto de entrenamiento
    
    print("\nPreparación de datos finalizada con éxito.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler

if __name__ == "__main__":
    # Ruta donde debes colocar el CSV descargado de Kaggle
    ruta_csv = "dataset.csv" 
    
    try:
        X_train, X_test, y_train, y_test, encoder, scaler = preparar_datos(ruta_csv)
        
        # Aquí ya tendrías los datos listos para introducirlos a tu modelo TensorFlow/Keras
        print("\n¡Listo! Las variables X_train, y_train ya pueden ser usadas en un modelo Sequential de Keras.")
        
    except Exception as e:
        print(f"Error: {e}")
