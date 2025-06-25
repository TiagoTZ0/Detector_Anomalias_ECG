Detector de Anomalías en ECG

# Guía de Instalación y Uso (Windows)

## 1. Requisitos
- Python 3.9.7 (¡No usar Python 3.10 o superior!)
- VS Code

## 2. Pasos de Instalación
1. Abrir VS Code
2. Abrir la carpeta del proyecto
3. Abrir terminal (Ctrl + `)
4. Ejecutar estos comandos en orden:

# Instalar dependencias del requirements.txt
pip install -r requirements.txt

# Ejecutar app
streamlit run main_app_ecg.py
```

## 3. Estructura del Proyecto

```
Proyecto_IA/
│
├── main_app_ecg.py      # Aplicación principal con interfaz Streamlit
├── entrenamiento.py     # Script de entrenamiento del modelo
├── preprocesado.py      # Funciones de procesamiento de señales
├── visualizacion.py     # Funciones de visualización
│
├── data/                # Datos de ejemplo
│   └── ptbdb_normal.csv
│
├── ECGsPARATEST/       # Datos de prueba
│   ├── anomalos/
│   └── saludables/
│
└── infogenerada/       # Archivos generados por el modelo
    ├── ecg_autoencoder_model.h5
    ├── ecg_scaler.pkl
    └── anomaly_threshold.json
```

## 4. Archivos Principales

### main_app_ecg.py
- Interfaz gráfica Streamlit
- Funciones de detección de anomalías
- Visualización de resultados

### entrenamiento.py
- Implementación del autoencoder
- Funciones de entrenamiento
- Guardado de modelo y umbrales

### preprocesado.py
- Filtrado de señales ECG
- Segmentación de datos
- Extracción de características

### visualizacion.py
- Gráficos de señales
- Visualización de anomalías
- Plots interactivos

## 5. Uso del Sistema

1. **Entrenamiento**
   - Seleccionar "Entrenamiento"
   - Cargar CSV con ECG normal
   - Configurar parámetros
   - Iniciar entrenamiento

2. **Detección**
   - Ir a "Detector de Anomalías"
   - Cargar ECG a analizar
   - Ver resultados y gráficos

## 6. Formato de Datos
El archivo CSV contiene señales ECG donde:
- Cada fila representa un latido cardíaco completo
- Cada fila contiene 160 valores numéricos que representan las mediciones del ECG
- Los valores están normalizados entre 0 y 1
- Se utiliza notación científica (e.g., 1.000000000000000000e+00)

Ejemplo del formato:
```csv
1.000000000000000000e+00,9.003241658210754395e-01,3.585899472236633301e-01,...
9.489833116531372070e-01,5.052650570869445801e-01,4.175744485110044479e-03,...
9.808584451675415039e-01,3.590487241744995117e-01,6.032482534646987915e-02,...

## 7. Archivos Generados
- ecg_autoencoder_model.h5: Modelo entrenado
- ecg_scaler.pkl: Normalizador
- anomaly_threshold.json: Umbral de anomalía

## 8. Consideraciones
- Usar solo Python 3.9.7
- Mantener estructura de carpetas
- Entrenar con ECGs normales primero
- Los archivos se generan automáticamente

## 9. Solución de Problemas
- Si hay error al activar el entorno, ejecutar:
  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## 10. Configuración Avanzada de Entrenamiento

### Parámetros Óptimos
- **Épocas**: 150 (recomendado)
  - Mínimo: 100
  - Máximo: 200
  - Aumentar si el modelo no converge

- **Tamaño de Segmento**: 160 (fijo)
  - No modificar para mantener compatibilidad
  - Optimizado para capturar un ciclo cardíaco completo

- **Solapamiento**: 0.5 (50%)
  - Mínimo: 0.25 (25%)
  - Máximo: 0.75 (75%)
  - Mayor solapamiento = mejor detección, pero más lento

- **Validación**: 0.2 (20%)
  - Rango recomendado: 0.15 - 0.25
  - No usar menos del 15% para evitar overfitting

### Hiperparámetros del Modelo
- Learning Rate: 0.001
- Batch Size: 32
- Early Stopping: 15 épocas
- Reduce LR on Plateau: factor 0.5, paciencia 5

### Umbrales de Detección
- Normal: percentil 95
- Estricto: percentil 99
- Permisivo: percentil 90

### Comando Recomendado
```bash
python entrenamiento.py --csv data/ptbdb_normal.csv --epochs 150 --segment_length 160 --overlap 0.5 --validation 0.2
```

### Consideraciones
1. **Calidad de Datos**
   - Usar ECGs limpios y bien etiquetados
   - Evitar señales con ruido excesivo
   - Asegurar normalización correcta

2. **Recursos Computacionales**
   - RAM mínima: 8GB
   - GPU recomendada para entrenamiento
   - Tiempo estimado: 15-30 minutos

3. **Monitoreo**
   - Observar la pérdida de validación
   - Verificar la convergencia del modelo
   - Revisar métricas guardadas en /infogenerada/

4. **Mejores Prácticas**
   - Entrenar con múltiples conjuntos de datos normales
   - Validar con datos conocidos antes de uso real
   - Mantener respaldo de modelos exitosos

   ### Comportamiento del Entrenamiento
. **Early Stopping**
   - El entrenamiento puede detenerse antes de las épocas máximas
   - Se detiene cuando no hay mejora en la validación
   - Típicamente ocurre entre las épocas 15-25
   
. **Señales de Convergencia**
   - Loss inicial: ~1.5-1.6
   - Loss final: ~0.2-0.3
   - Val_loss final: ~0.8-0.9
   
. **Learning Rate**
   - Inicial: 0.001
   - Reducción automática: 50% cada 5 épocas sin mejora
   - Mínimo: 0.000001

. **Criterios de Parada**
   - No mejora en val_loss durante 15 épocas
   - Learning rate llega al mínimo
   - Se alcanza performance óptimo


### Archivos de Configuración
El archivo `anomaly_threshold.json` contiene:
```json
{
    "threshold_normal": 0.95,    // Umbral estándar (percentil 95)
    "threshold_strict": 0.99,    // Umbral estricto (percentil 99)
    "threshold_lenient": 0.90,   // Umbral permisivo (percentil 90)
    "mean_error": 0.123,         // Error medio del conjunto de validación
    "std_error": 0.456,         // Desviación estándar del error
    "p90_error": 0.789,         // Percentil 90 del error
    "p95_error": 0.901,         // Percentil 95 del error
    "p99_error": 0.999          // Percentil 99 del error
}