import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
import json
import streamlit as st
from sklearn.preprocessing import StandardScaler
import tempfile
import pickle

# Importar módulos personalizados
from preprocesado import bandpass_filter, extract_features, segment_signal
from visualizacion import plot_anomaly_results
from entrenamiento import train_model  # Importa la función directamente

def get_custom_objects():
    """Crear y retornar objetos personalizados para el modelo"""
    def custom_loss():
        """Función de pérdida personalizada para el autoencoder"""
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
    loss_fn = custom_loss()
    return {
        'custom_loss': loss_fn,
        'loss': loss_fn
    }

custom_objects = get_custom_objects()

def detect_anomaly_in_ecg(file_path, model_path="infogenerada/ecg_autoencoder_model.h5", threshold=None):
    """Detectar anomalías en un archivo ECG (procesando toda la señal)"""
    try:
        if not os.path.exists(model_path):
            st.error(f"No se encuentra el modelo en {model_path}")
            return None

        # --- Leer el archivo CSV y extraer solo columnas numéricas ---
        if file_path.endswith('.csv'):
            try:
                data = pd.read_csv(file_path)
                # Seleccionar solo columnas numéricas
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    st.error("El archivo CSV no contiene columnas numéricas.")
                    return None
                raw_signal = numeric_data.iloc[:, 0].values.astype(float)
                if len(numeric_data.columns) > 1:
                    st.warning("El archivo tiene múltiples columnas numéricas. Se usará la primera columna numérica.")
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {str(e)}")
                return None
        else:
            st.error("El archivo debe ser CSV.")
            return None

        filtered_signal = bandpass_filter(raw_signal)
        scaler_path = 'infogenerada/ecg_scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                n_features = getattr(scaler, 'n_features_in_', 1)
                if n_features > 1:
                    segment_length = n_features
                    segments = segment_signal(filtered_signal, segment_length=segment_length, overlap=0.75)
                    if len(segments) == 0:
                        st.error("No se pudieron crear segmentos de la señal para normalizar.")
                        return None
                    normalized_segments = scaler.transform(segments)
                    normalized_signal = normalized_segments.flatten()
                else:
                    normalized_signal = scaler.transform(filtered_signal.reshape(-1, 1)).flatten()
            except Exception as e:
                st.error(f"Error al cargar el scaler: {str(e)}")
                return None
        else:
            st.error("No se encuentra el archivo del scaler.")
            return None

        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            model.compile(
                optimizer='adam',
                loss=custom_objects['custom_loss'],
                metrics=['mae', 'mse']
            )
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None

        with open('infogenerada/anomaly_threshold.json', 'r') as f:
            thresholds = json.load(f)
        threshold_value = threshold if threshold is not None else thresholds['threshold_normal']
        input_shape = model.input_shape[1]
        segments = segment_signal(normalized_signal, segment_length=input_shape, overlap=0.75)
        if len(segments) == 0:
            st.error("No se pudieron crear segmentos de la señal.")
            return None

        # Procesar todos los segmentos en batch para máxima velocidad
        reconstructed_segments = model.predict(segments.reshape(-1, input_shape, 1), verbose=0)
        # FIX: broadcasting, quitar la última dimensión de reconstructed_segments
        reconstructed_segments = reconstructed_segments.squeeze(-1)
        segment_errors = np.mean(np.square(segments - reconstructed_segments), axis=1)
        max_error = np.max(segment_errors)
        mean_error = np.mean(segment_errors)
        base_threshold = threshold if threshold is not None else thresholds['threshold_normal']
        adjusted_threshold = base_threshold * 1.5
        is_anomaly = max_error > (mean_error * 3)
        is_severe = max_error > (mean_error * 5)
        is_mild = (max_error > (mean_error * 2)) and not is_severe and not is_anomaly
        severity_base = thresholds['threshold_strict'] * 1.5
        severity_score = min(1.0, max_error / severity_base)
        results = {
            'is_anomaly': bool(is_anomaly),
            'is_severe_anomaly': bool(is_severe),
            'is_mild_anomaly': bool(is_mild),
            'severity_score': float(severity_score),
            'max_mse': float(max_error),
            'mean_mse': float(mean_error),
            'threshold': float(adjusted_threshold),
            'threshold_strict': float(thresholds['threshold_strict'] * 1.5),
            'threshold_lenient': float(thresholds['threshold_lenient'] * 1.2),
            'worst_segment': segments[np.argmax(segment_errors)].tolist(),
            'worst_reconstructed': reconstructed_segments[np.argmax(segment_errors)].tolist(),
            'segment_errors': segment_errors.tolist()  # Para graficar el error cuadrático
        }
        return results
    except Exception as e:
        st.error(f"Error en la detección: {str(e)}")
        st.exception(e)
        return None

def main():
    st.set_page_config(page_title="Detector de Anomalias en ECGs", layout="wide")
    st.title("Detector de Anomalías en ECG")
    st.markdown("""
    Esta aplicación permite detectar anomalías en señales de electrocardiograma (ECG) 
    utilizando técnicas de aprendizaje automático con autoencoder. 
    """)
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Ir a", ["Inicio", "Detector de Anomalías", "Entrenamiento", "Acerca de"])
    if page == "Inicio":
        st.markdown("""
        ## Bienvenido al Detector de Anomalías en ECG
        
        Esta herramienta te permite:
        1. **Detectar anomalías** en señales ECG
        2. **Entrenar** el modelo con tus propios datos normales
        3. **Visualizar** los resultados con gráficos detallados
        """)
        st.info("💡 Recomendamos primero entrenar el modelo con datos de ECG normales.")
        if os.path.exists("infogenerada/ecg_autoencoder_model.h5"):
            st.success("✅ Modelo entrenado detectado y listo para usar.")
        else:
            st.warning("⚠️ No se ha detectado un modelo entrenado. Por favor, ve a Entrenamiento.")
    elif page == "Detector de Anomalías":
        st.header("Detector de Anomalías en ECG")
        if not os.path.exists("infogenerada/ecg_autoencoder_model.h5"):
            st.warning("⚠️ No se ha encontrado un modelo entrenado. Por favor, entrena primero.")
            return
        st.success("✅ Modelo cargado y listo para detectar anomalías.")
        with st.expander("⚙️ Configuración avanzada"):
            try:
                with open("infogenerada/anomaly_threshold.json", 'r') as f:
                    threshold_data = json.load(f)
                    default_threshold = threshold_data['threshold_normal']
            except:
                default_threshold = 0.01
            custom_threshold = st.slider(
                "Umbral de anomalía personalizado", 
                min_value=float(default_threshold * 0.5), 
                max_value=float(default_threshold * 3.0), 
                value=float(default_threshold),
                step=float(default_threshold * 0.05),
                format="%.6f"
            )
            st.info(f"""
            Umbral original: {default_threshold:.6f}
            Umbral actual: {custom_threshold:.6f}
            """)
        uploaded_file = st.file_uploader("Cargar archivo CSV con datos de ECG", type=["csv"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            try:
                with st.spinner('Analizando señal ECG...'):
                    results = detect_anomaly_in_ecg(temp_path, threshold=custom_threshold)
                    if results:
                        plot_anomaly_results(results, temp_path)
                        download_results = {
                            k: float(v) if isinstance(v, (np.float32, np.float64)) else
                               bool(v) if isinstance(v, np.bool_) else v
                            for k, v in results.items()
                            if not isinstance(v, (np.ndarray, list))
                        }
                        results_json = json.dumps(download_results)
                        st.download_button(
                            "📥 Descargar resultados (JSON)",
                            results_json,
                            "ecg_anomaly_results.json",
                            "application/json"
                        )
            except Exception as e:
                st.error(f"Error en el procesamiento: {str(e)}")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    elif page == "Entrenamiento":
        st.header("Entrenamiento del Detector")
        st.markdown("""
        ### Instrucciones
        1. Sube un archivo CSV con datos de ECG **normal**
        2. Configura los parámetros
        3. Inicia el entrenamiento
        """)
        with st.expander("⚙️ Configuración avanzada"):
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Épocas", 10, 200, 150, 10)
                validation = st.slider("Validación", 0.1, 0.4, 0.2, 0.05)
            with col2:
                segment_length = st.number_input("Tamaño segmentos", 100, 1000, 160, 50)
                overlap = st.slider("Solapamiento", 0.0, 0.9, 0.5, 0.1)
        uploaded_train_file = st.file_uploader("Cargar archivo CSV de ECG normal", type=["csv"])
        if uploaded_train_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_train_file.getvalue())
                temp_train_path = tmp_file.name
            if st.button("🚀 Iniciar Entrenamiento"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                def progress_callback(msg, pct):
                    status_text.write(msg)
                    if pct is not None:
                        progress_bar.progress(min(100, pct))
                try:
                    model, history = train_model(
                        temp_train_path,
                        epochs=epochs,
                        validation_split=validation,
                        segment_length=segment_length,
                        overlap=overlap,
                        progress_callback=progress_callback
                    )
                    status_text.success("✅ Entrenamiento completado exitosamente!")
                    progress_bar.progress(100)
                except Exception as e:
                    status_text.error(f"❌ Error en el entrenamiento: {e}")
                    progress_bar.progress(100)
    elif page == "Acerca de":
        st.header("Acerca del Detector")
        st.markdown("""
        ### Funcionamiento
        
        Utiliza **aprendizaje profundo** para detectar anomalías en ECGs mediante:
        1. **Autoencoder:** Aprende a reconstruir señales normales
        2. **Umbral estadístico:** Basado en errores de reconstrucción
        3. **Detección:** Compara errores con umbrales establecidos
        
        ### ⚠️ Advertencias
        - NO es un dispositivo médico certificado
        - Requiere validación por profesionales médicos
        - El rendimiento depende de los datos de entrenamiento
        
        ### 🔧 Características
        - Filtrado pasa banda
        - Análisis morfológico
        - Autoencoder convolucional
        - Data augmentation
        
        ### 📚 Referencias
        - "Anomaly Detection in ECG Using Deep Learning" (2020)
        - "ECG Signal Quality Assessment" (2021)
        
        Versión 1.0
        """)

if __name__ == "__main__":
    main()