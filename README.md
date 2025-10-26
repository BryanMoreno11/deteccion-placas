# Sistema de Detección de Placas Vehiculares

Sistema web completo para la detección automática de placas vehiculares usando visión por computadora y OCR, con consulta de datos del vehículo mediante API externa.

## 🚀 Características

- **Detección de Placas**: Modelo YOLO en formato ONNX para detectar placas en imágenes
- **Extracción OCR**: EasyOCR para extraer el número de placa
- **Consulta de Datos**: Integración con API RegCheck para obtener información del vehículo
- **Interfaz Web Moderna**: Frontend responsive con Bootstrap 5
- **Arquitectura Separada**: Backend (Flask API) y Frontend (Flask Web) en puertos diferentes
- **Validaciones Inteligentes**: Manejo de múltiples placas, placas no detectadas, y baja confianza

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Navegador web moderno (Chrome, Firefox, Edge)

## 🔧 Instalación

1. **Clonar o descargar el proyecto**

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv .venv
```

3. **Activar el entorno virtual**

Windows:
```bash
.venv\Scripts\activate
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 🏗️ Estructura del Proyecto

```
deteccionPlacas/
├── backend/                    # Backend API
│   ├── __init__.py            # Configuración de Flask backend
│   └── routes/
│       └── plate_routes.py    # Endpoints de la API
├── frontend/                   # Frontend Web
│   ├── __init__.py            # Configuración de Flask frontend
│   └── gestionPlacas/
│       ├── routes.py          # Rutas del frontend
│       ├── templates/
│       │   └── gestion_placas.html
│       └── static/
│           ├── css/
│           │   └── styles.css
│           └── js/
│               └── app.js
├── services/                   # Servicios de procesamiento
│   ├── licencePlateDetection.py
│   ├── extractPlate.py
│   ├── extractPlatePipeline.py
│   └── getPlateData.py
├── models/                     # Modelos de ML
│   └── modelo_placas.onnx
├── backend_run.py             # Punto de entrada del backend
├── frontend_run.py            # Punto de entrada del frontend
├── settings.py                # Configuración global
└── requirements.txt           # Dependencias
```

## 🚀 Uso

### Iniciar el Backend (Puerto 5000)

Abre una terminal y ejecuta:

```bash
python backend_run.py
```

El backend estará disponible en: `http://localhost:5000`

### Iniciar el Frontend (Puerto 5001)

Abre **otra terminal** y ejecuta:

```bash
python frontend_run.py
```

El frontend estará disponible en: `http://localhost:5001`

### Acceder a la Aplicación

1. Abre tu navegador web
2. Ve a: `http://localhost:5001`
3. Sube una imagen de un vehículo con placa visible o toma una foto con tu cámara
4. Haz clic en "Procesar Imagen"
5. Visualiza los resultados:
   - Imagen recortada de la placa
   - Número de placa detectado
   - Datos del vehículo (año, tipo, marca, modelo)

## 📡 API Endpoints

### Backend API (Puerto 5000)

#### POST `/api/process-plate`
Procesa una imagen y retorna la placa detectada con datos del vehículo.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (archivo de imagen)

**Response (Éxito):**
```json
{
  "status": "success",
  "plate_image": "base64_encoded_image",
  "plate_number": "ABC1234",
  "confidence": 0.95,
  "detection_confidence": 0.98,
  "vehicle_data": {
    "year": "2020",
    "type": "AUTOMOVIL",
    "subtype": "USO PARTICULAR",
    "make": "SUZUKI",
    "model": "SCROSS AC 1.6 5P 4X2 TM",
    "description": "SUZUKI SCROSS AC 1.6 5P 4X2 TM",
    "image_url": "http://..."
  },
  "vehicle_status": "success",
  "message": "Placa ABC1234 procesada correctamente"
}
```

**Response (Error - No Placa Detectada):**
```json
{
  "status": "no_plate_detected",
  "message": "No se detectó ninguna placa. Por favor, intente:\n• Acercarse más al vehículo\n• Asegurar buena iluminación\n• Centrar la placa en la imagen"
}
```

**Response (Error - Múltiples Placas):**
```json
{
  "status": "multiple_plates_detected",
  "message": "Se detectaron 2 placas en la imagen.\nPor favor, tome una nueva foto donde solo aparezca la placa que desea consultar."
}
```

#### GET `/api/health`
Verifica el estado del servidor.

**Response:**
```json
{
  "status": "ok",
  "message": "Backend de detección de placas funcionando correctamente"
}
```

## 🎨 Características de la Interfaz

- **Diseño Responsive**: Se adapta a móviles, tablets y escritorio
- **Drag & Drop**: Arrastra imágenes directamente al área de carga
- **Captura de Cámara**: Toma fotos directamente desde el navegador
- **Animaciones Suaves**: Transiciones y efectos visuales elegantes
- **Feedback Visual**: Indicadores de progreso y mensajes claros
- **Validaciones**: Manejo de errores con mensajes descriptivos

## 🔍 Validaciones Implementadas

El sistema maneja los siguientes casos:

1. **Sin placa detectada**: Sugiere mejorar iluminación y acercarse más
2. **Múltiples placas**: Solicita tomar nueva foto con una sola placa
3. **Texto no detectado**: Muestra la placa recortada pero indica que no se pudo leer
4. **Baja confianza**: Muestra el resultado pero advierte sobre la confianza baja
5. **Formato inválido**: Indica que el texto detectado no coincide con formato de placa típico
6. **Datos no encontrados**: Muestra la placa pero indica que no hay datos del vehículo

## ⚙️ Configuración

### Cambiar el Username de RegCheck

Edita el archivo `backend/routes/plate_routes.py`:

```python
REGCHECK_USERNAME = "TuUsuarioAqui"
```

### Ajustar Umbrales de Confianza

Edita el archivo `settings.py`:

```python
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # Umbral de detección (0-1)
OCR_CONFIDENCE_THRESHOLD = 0.5        # Umbral de OCR (0-1)
IOU_THRESHOLD = 0.45                  # Umbral de IoU para NMS
```

### Cambiar Puertos

**Backend** - Edita `backend_run.py`:
```python
app.run(debug=True, host="0.0.0.0", port=5000)  # Cambia el puerto aquí
```

**Frontend** - Edita `frontend_run.py`:
```python
app.run(debug=True, host="0.0.0.0", port=5001)  # Cambia el puerto aquí
```

**JavaScript** - Edita `frontend/gestionPlacas/static/js/app.js`:
```javascript
const API_BASE_URL = 'http://localhost:5000/api';  // Actualiza si cambias el puerto del backend
```

## 🐛 Solución de Problemas

### Error: "No se pudo acceder a la cámara"
- Verifica que el navegador tenga permisos para acceder a la cámara
- Usa HTTPS o localhost (las cámaras no funcionan en HTTP en producción)

### Error: "Error de conexión con el servidor"
- Verifica que el backend esté ejecutándose en el puerto 5000
- Revisa la consola del backend para ver errores

### Error: "No se pudo cargar el modelo ONNX"
- Verifica que el archivo `models/modelo_placas.onnx` exista
- Asegúrate de que onnxruntime esté instalado correctamente

### La detección no funciona bien
- Asegúrate de que la imagen tenga buena iluminación
- La placa debe estar visible y enfocada
- Evita imágenes con múltiples vehículos

## 📝 Notas Importantes

- El backend debe estar ejecutándose antes de usar el frontend
- Las imágenes se procesan en el servidor, no se almacenan permanentemente
- La API de RegCheck requiere un username válido
- El sistema funciona mejor con imágenes de buena calidad

## 🔒 Seguridad

- El tamaño máximo de archivo es 16MB
- Solo se aceptan imágenes JPG y PNG
- CORS está habilitado para desarrollo (ajustar en producción)
- No se almacenan imágenes en el servidor

## 📄 Licencia

Este proyecto es de uso educativo y de demostración.

## 👨‍💻 Autor

Bryan Moreno

---

**¡Disfruta usando el sistema de detección de placas vehiculares!** 🚗📸
