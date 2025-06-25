import os
import shutil

def setup_environment():
    """Preparar el entorno para el entrenamiento"""
    
    print("🔄 Configurando entorno...")
    
    # 1. Limpiar directorio infogenerada si existe
    if os.path.exists('infogenerada'):
        print("🗑️ Limpiando directorio infogenerada...")
        shutil.rmtree('infogenerada')
    
    # 2. Crear directorio limpio
    print("📁 Creando nuevo directorio infogenerada...")
    os.makedirs('infogenerada', exist_ok=True)
    
    # 3. Verificar estructura de directorios
    required_dirs = ['data', 'ECGsPARATEST/anomalos', 'ECGsPARATEST/saludables']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"📁 Creando directorio: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Configuración completada!")

if __name__ == "__main__":
    setup_environment()