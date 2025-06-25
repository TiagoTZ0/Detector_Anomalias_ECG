import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from scipy.signal import butter, filtfilt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
from sklearn.preprocessing import StandardScaler
import json
import argparse
from sklearn.model_selection import train_test_split
from preprocesado import bandpass_filter, segment_signal

def create_improved_autoencoder(segment_length):
    """Crear autoencoder optimizado para ECG"""
    input_layer = Input(shape=(segment_length, 1))
    
    # Encoder
    x = layers.Conv1D(32, 3, padding='same')(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Bottleneck
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    encoded = BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv1D(128, 3, padding='same')(encoded)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.UpSampling1D(2)(x)
    
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.UpSampling1D(2)(x)
    
    x = layers.Conv1D(32, 3, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = layers.UpSampling1D(2)(x)
    
    output_layer = layers.Conv1D(1, 1, activation='tanh')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

def custom_loss():
    """Función de pérdida personalizada para el autoencoder"""
    mse = MeanSquaredError()
    
    @tf.function
    def loss(y_true, y_pred):
        # MSE básico
        mse_loss = mse(y_true, y_pred)
        
        # Calcular gradientes usando diferencias finitas
        # No usar GradientTape aquí
        grad_true = tf.reduce_mean(tf.square(y_true[:, 1:] - y_true[:, :-1]), axis=1)
        grad_pred = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]), axis=1)
        gradient_loss = tf.reduce_mean(tf.square(grad_true - grad_pred))
        
        return mse_loss + 0.1 * gradient_loss
    
    # Asignar un nombre a la función de pérdida
    loss.__name__ = 'custom_loss'
    return loss

# Registrar la función de pérdida personalizada
custom_objects = {
    'custom_loss': custom_loss(),  # Nota: llamamos a la función aquí
    'loss': custom_loss()  # Añadir esta línea por compatibilidad
}

def train_model(csv_path, epochs=150, multiplier=1.2, validation_split=0.2, segment_length=160, overlap=0.5):
    """Entrenar el modelo"""
    try:
        # Cargar y preparar datos
        print("Cargando datos...")
        data = pd.read_csv(csv_path)
        raw_signal = data[data.columns[0]].values
        
        # Preprocesar señal
        print("Preprocesando señal...")
        filtered_signal = bandpass_filter(raw_signal)
        scaler = StandardScaler()
        normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
        
        # Crear directorio si no existe
        os.makedirs("infogenerada", exist_ok=True)
        
        # Guardar scaler
        print("Guardando scaler...")
        with open('infogenerada/ecg_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Segmentar señal
        print("Segmentando señal...")
        segments = segment_signal(normalized_signal, segment_length=segment_length, overlap=overlap)
        
        # Dividir datos
        print("Dividiendo datos en train/val...")
        train_segments, val_segments = train_test_split(segments, test_size=validation_split, random_state=42)
        train_data = train_segments.reshape(-1, segment_length, 1)
        val_data = val_segments.reshape(-1, segment_length, 1)
        
        # Crear y compilar modelo
        print("Creando modelo...")
        model = create_improved_autoencoder(segment_length)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=custom_loss(),
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        print("Configurando callbacks...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Entrenar
        print("Iniciando entrenamiento...")
        history = model.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=32,
            validation_data=(val_data, val_data),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular umbrales
        print("Calculando umbrales de detección...")
        reconstruction_errors = []
        for segment in val_data:
            reconstructed = model.predict(segment.reshape(1, segment_length, 1), verbose=0)[0]
            mse = np.mean(np.square(segment - reconstructed))
            reconstruction_errors.append(mse)
        
        # Calcular múltiples umbrales
        threshold_data = {
            'threshold': float(np.percentile(reconstruction_errors, 95)),  # Para compatibilidad
            'threshold_normal': float(np.percentile(reconstruction_errors, 95)),
            'threshold_strict': float(np.percentile(reconstruction_errors, 99)),
            'threshold_lenient': float(np.percentile(reconstruction_errors, 90)),
            'mean_error': float(np.mean(reconstruction_errors)),
            'std_error': float(np.std(reconstruction_errors)),
            'p90_error': float(np.percentile(reconstruction_errors, 90)),
            'p95_error': float(np.percentile(reconstruction_errors, 95)),
            'p99_error': float(np.percentile(reconstruction_errors, 99))
        }
        
        # Guardar modelo y métricas
        print("Guardando archivos generados...")
        model.save('infogenerada/ecg_autoencoder_model.h5')
        
        with open('infogenerada/anomaly_threshold.json', 'w') as f:
            json.dump(threshold_data, f, indent=4)
        
        with open('infogenerada/training_metrics.json', 'w') as f:
            metrics = {
                'history': {k: [float(x) for x in v] for k, v in history.history.items()},
                'final_metrics': {
                    'final_loss': float(history.history['loss'][-1]),
                    'best_val_loss': float(min(history.history['val_loss'])),
                    'epochs_trained': len(history.history['loss'])
                }
            }
            json.dump(metrics, f, indent=4)
        
        print("Entrenamiento completado y archivos guardados en 'infogenerada/'")
        return model, history

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de anomalías ECG')
    parser.add_argument('--csv', type=str, required=True, help='Ruta al archivo CSV con datos ECG')
    parser.add_argument('--epochs', type=int, default=150, help='Número de épocas')
    parser.add_argument('--multiplier', type=float, default=1.2, help='Multiplicador para el umbral')
    parser.add_argument('--validation', type=float, default=0.2, help='Proporción de validación')
    parser.add_argument('--segment_length', type=int, default=160, help='Longitud de los segmentos')
    parser.add_argument('--overlap', type=float, default=0.5, help='Solapamiento entre segmentos')
    
    args = parser.parse_args()
    train_model(
        args.csv,
        epochs=args.epochs,
        multiplier=args.multiplier,
        validation_split=args.validation,
        segment_length=args.segment_length,
        overlap=args.overlap
    )