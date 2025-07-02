import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras import layers, Input
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

def create_flexible_autoencoder(segment_length):
    input_layer = Input(shape=(segment_length, 1))
    x = layers.Conv1D(32, 3, padding='same')(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)

    # Solo una capa de pooling/upsampling si el tamaño no es múltiplo de 4
    if segment_length % 4 == 0:
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = Dropout(0.3)(x)
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)
        encoded = x
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
    else:
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = Dropout(0.3)(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)
        encoded = x
        x = layers.Conv1D(64, 3, padding='same')(encoded)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(32, 3, padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)

    x = layers.Conv1D(1, 1, activation='tanh', padding='same')(x)
    # Ajuste final: recorta o rellena para igualar tamaño de entrada
    x = layers.Lambda(lambda y: y[:, :segment_length, :])(x)
    return Model(inputs=input_layer, outputs=x)

def custom_loss():
    mse = MeanSquaredError()
    @tf.function
    def loss(y_true, y_pred):
        mse_loss = mse(y_true, y_pred)
        grad_true = tf.reduce_mean(tf.square(y_true[:, 1:] - y_true[:, :-1]), axis=1)
        grad_pred = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]), axis=1)
        gradient_loss = tf.reduce_mean(tf.square(grad_true - grad_pred))
        return mse_loss + 0.1 * gradient_loss
    loss.__name__ = 'custom_loss'
    return loss

custom_objects = {
    'custom_loss': custom_loss(),
    'loss': custom_loss()
}

def train_model(
    csv_path, epochs=150, multiplier=1.2, validation_split=0.2,
    segment_length=160, overlap=0.5, progress_callback=None
):
    try:
        def update(msg, pct=None):
            if progress_callback:
                progress_callback(msg, pct)
            print(msg, flush=True)

        update("Cargando datos PTBDB o señal larga...", 0)
        data = pd.read_csv(csv_path, header=None)
        if data.shape[1] > 10 and data.shape[0] > 1:
            update("Detectado dataset tipo PTBDB (cada fila es un segmento)", 5)
            X = data.iloc[:, :-1].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            os.makedirs("infogenerada", exist_ok=True)
            update("Guardando scaler...", 10)
            with open('infogenerada/ecg_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            X_scaled = X_scaled.reshape(-1, X.shape[1], 1)
            train_data, val_data = train_test_split(X_scaled, test_size=validation_split, random_state=42)
            segment_len = X.shape[1]
        else:
            update("Detectado señal larga (una sola fila o columna)", 5)
            raw_signal = data.values.flatten()
            filtered_signal = bandpass_filter(raw_signal)
            scaler = StandardScaler()
            normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
            os.makedirs("infogenerada", exist_ok=True)
            update("Guardando scaler...", 10)
            with open('infogenerada/ecg_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            update("Segmentando señal...", 15)
            segments = segment_signal(normalized_signal, segment_length=segment_length, overlap=overlap)
            train_segments, val_segments = train_test_split(segments, test_size=validation_split, random_state=42)
            train_data = train_segments.reshape(-1, segment_length, 1)
            val_data = val_segments.reshape(-1, segment_length, 1)
            segment_len = segment_length

        update("Creando modelo...", 20)
        model = create_flexible_autoencoder(segment_len)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=custom_loss(),
            metrics=['mae', 'mse']
        )
        update("Configurando callbacks...", 25)
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
        update("Iniciando entrenamiento...", 30)
        # Callback para progreso de epochs
        class StreamlitProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pct = 30 + int(60 * (epoch + 1) / epochs)
                msg = f"Época {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}"
                update(msg, pct)

        history = model.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=32,
            validation_data=(val_data, val_data),
            callbacks=callbacks + [StreamlitProgress()],
            verbose=0
        )
        update("Calculando umbrales de detección...", 95)
        reconstruction_errors = []
        for segment in val_data:
            reconstructed = model.predict(segment.reshape(1, segment_len, 1), verbose=0)[0]
            mse = np.mean(np.square(segment - reconstructed))
            reconstruction_errors.append(mse)
        threshold_data = {
            'threshold': float(np.percentile(reconstruction_errors, 95)),
            'threshold_normal': float(np.percentile(reconstruction_errors, 95)),
            'threshold_strict': float(np.percentile(reconstruction_errors, 99)),
            'threshold_lenient': float(np.percentile(reconstruction_errors, 90)),
            'mean_error': float(np.mean(reconstruction_errors)),
            'std_error': float(np.std(reconstruction_errors)),
            'p90_error': float(np.percentile(reconstruction_errors, 90)),
            'p95_error': float(np.percentile(reconstruction_errors, 95)),
            'p99_error': float(np.percentile(reconstruction_errors, 99))
        }
        update("Guardando archivos generados...", 98)
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
        update("Entrenamiento completado y archivos guardados en 'infogenerada/'", 100)
        return model, history
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error durante el entrenamiento: {str(e)}", 100)
        print(f"Error durante el entrenamiento: {str(e)}", flush=True)
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