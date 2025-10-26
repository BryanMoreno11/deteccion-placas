from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "test"
TEST_IMAGES_DIR = TEST_DIR / "test_images"
OUTPUT_DIR = TEST_DIR / "test_detection_output"

# Modelo de detección de placas
PLATE_DETECTION_MODEL = MODELS_DIR / "modelo_placas.onnx"

# Parámetros del modelo
DETECTION_CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 800  # Tamaño usado en entrenamiento
IOU_THRESHOLD = 0.45 
# Crear directorios si no existen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#Parámetros OCR
OCR_CONFIDENCE_THRESHOLD = 0.5
API_USERNAME="prueba1234"