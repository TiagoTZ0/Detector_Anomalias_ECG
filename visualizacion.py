import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_anomaly_results(results, file_path):
    """Visualiza los resultados del análisis de anomalías de manera detallada"""
    if not results:
        return
    
    # Extraer información clave
    is_anomaly = results['is_anomaly']
    is_severe = results.get('is_severe_anomaly', False)
    is_mild = results.get('is_mild_anomaly', False)
    max_error = results['max_mse']
    threshold = results['threshold']
    
    # Determinar el tipo de resultado y color
    if is_severe:
        result_type = "⚠️ ANOMALÍA GRAVE DETECTADA"
        color = "darkred"
    elif is_anomaly:
        result_type = "⚠️ ANOMALÍA DETECTADA"
        color = "red"
    elif is_mild:
        result_type = "🟡 ANOMALÍA LEVE DETECTADA"
        color = "orange"
    else:
        result_type = "✅ ECG NORMAL (SALUDABLE)"
        color = "green"
    
    # Mostrar resultado principal
    st.markdown(f"## {result_type}")
    
    # Mostrar métricas en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Error Máximo", 
            f"{max_error:.6f}",
            f"{((max_error/threshold - 1) * 100):.1f}% del umbral"
        )
    
    with col2:
        st.metric(
            "Umbral Normal",
            f"{threshold:.6f}"
        )
    
    with col3:
        confidence = (1 - (max_error/results['threshold_strict'])) * 100
        confidence = max(0, min(100, confidence))
        st.metric(
            "Confianza de Normalidad",
            f"{confidence:.1f}%"
        )
    
    # Mostrar detalles adicionales en un expander
    with st.expander("📊 Ver detalles técnicos"):
        st.json({
            'max_error': float(max_error),
            'mean_error': float(results['mean_mse']),
            'threshold_normal': float(threshold),
            'threshold_strict': float(results['threshold_strict']),
            'threshold_lenient': float(results['threshold_lenient']),
            'severity_score': float(results['severity_score'])
        })
    
    # Visualizar segmento con mayor error
    if 'worst_segment' in results and 'worst_reconstructed' in results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original vs reconstruido
        worst_segment = np.array(results['worst_segment'])
        worst_reconstructed = np.array(results['worst_reconstructed'])
        x = np.arange(len(worst_segment))
        
        ax1.plot(x, worst_segment, label='Original', color='blue', alpha=0.7)
        ax1.plot(x, worst_reconstructed, label='Reconstruido', color=color, alpha=0.7)
        ax1.set_title('Segmento con Mayor Error de Reconstrucción')
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('Amplitud')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot del error
        error = np.square(worst_segment - worst_reconstructed)
        ax2.plot(x, error, color=color, alpha=0.7)
        ax2.axhline(y=threshold, color='red', linestyle='--', label='Umbral')
        ax2.set_title('Error Cuadrático por Muestra')
        ax2.set_xlabel('Muestras')
        ax2.set_ylabel('Error²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Mostrar advertencias según el tipo de anomalía
    if is_severe:
        st.error("⚠️ ATENCIÓN: Se ha detectado una anomalía grave que requiere revisión médica inmediata.")
    elif is_anomaly:
        st.warning("⚠️ Se ha detectado una anomalía que debería ser evaluada por un profesional.")
    elif is_mild:
        st.info("ℹ️ Se ha detectado una anomalía leve. Se recomienda seguimiento.")
    else:
        st.success("✅ No se han detectado anomalías significativas.")