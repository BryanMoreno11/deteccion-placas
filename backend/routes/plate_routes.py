from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
from pathlib import Path
import tempfile
import os
from settings import API_USERNAME

from services.extractPlatePipeline import PlateRecognitionPipeline, PipelineStatus
from services.getPlateData import get_plate_data

bp_plate = Blueprint("plate", __name__)

# Inicializar pipeline una sola vez
print("🚀 Inicializando pipeline de reconocimiento de placas...")
pipeline = PlateRecognitionPipeline()
print("✅ Pipeline listo para procesar solicitudes")

# Username para la API de RegCheck
REGCHECK_USERNAME = API_USERNAME


def image_to_base64(image_array):
    """Convierte un array numpy de imagen a base64"""
    _, buffer = cv2.imencode('.jpg', image_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@bp_plate.route("/process-plate", methods=["POST"])
def process_plate():
    """
    Endpoint principal para procesar una imagen de placa
    
    Recibe: imagen en formato multipart/form-data
    Retorna: {
        status: success/error,
        plate_image: base64 (imagen recortada de la placa),
        plate_number: string,
        vehicle_data: object,
        message: string
    }
    """
    try:
        # Validar que se envió un archivo
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No se envió ninguna imagen'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No se seleccionó ningún archivo'
            }), 400
        
        # Leer la imagen
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'No se pudo decodificar la imagen. Asegúrese de enviar un formato válido (JPG, PNG)'
            }), 400
        
        # Guardar temporalmente la imagen para procesarla
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, image)
        
        try:
            # PASO 1: Procesar imagen con el pipeline (detección + OCR)
            print(f"📸 Procesando imagen...")
            result = pipeline.process_image(temp_path)
            
            # Verificar si la detección fue exitosa
            if result.status == PipelineStatus.NO_PLATE_DETECTED:
                return jsonify({
                    'status': 'no_plate_detected',
                    'message': result.message
                }), 200
            
            elif result.status == PipelineStatus.MULTIPLE_PLATES_DETECTED:
                return jsonify({
                    'status': 'multiple_plates_detected',
                    'message': result.message
                }), 200
            
            elif result.status == PipelineStatus.NO_TEXT_DETECTED:
                # Tenemos la placa recortada pero no se pudo leer el texto
                plate_image_base64 = None
                if result.detection_result and result.detection_result.plate_image is not None:
                    plate_image_base64 = image_to_base64(result.detection_result.plate_image)
                
                return jsonify({
                    'status': 'no_text_detected',
                    'message': result.message,
                    'plate_image': plate_image_base64
                }), 200
            
            elif result.status in [PipelineStatus.LOW_CONFIDENCE, PipelineStatus.INVALID_FORMAT]:
                # Tenemos placa y texto pero con advertencias
                plate_image_base64 = None
                if result.detection_result and result.detection_result.plate_image is not None:
                    plate_image_base64 = image_to_base64(result.detection_result.plate_image)
                
                return jsonify({
                    'status': result.status.value,
                    'message': result.message,
                    'plate_image': plate_image_base64,
                    'plate_number': result.plate_text,
                    'confidence': result.confidence
                }), 200
            
            elif result.status != PipelineStatus.SUCCESS:
                return jsonify({
                    'status': 'error',
                    'message': result.message
                }), 500
            
            # ÉXITO: Tenemos placa detectada y texto extraído
            plate_number = result.plate_text
            print(f"✅ Placa detectada: {plate_number}")
            
            # Convertir imagen de placa a base64
            plate_image_base64 = None
            if result.detection_result and result.detection_result.plate_image is not None:
                plate_image_base64 = image_to_base64(result.detection_result.plate_image)
            
            # PASO 2: Obtener datos del vehículo desde la API
            print(f"🔍 Consultando datos del vehículo...")
            vehicle_response = get_plate_data(
                plate_number=plate_number,
                username=REGCHECK_USERNAME,
                timeout=10.0,
                raw=False
            )
            
            # Preparar respuesta
            response_data = {
                'status': 'success',
                'plate_image': plate_image_base64,
                'plate_number': plate_number,
                'confidence': result.confidence,
                'detection_confidence': result.detection_confidence,
                'vehicle_data': None,
                'vehicle_status': vehicle_response['status'],
                'message': f'Placa {plate_number} procesada correctamente'
            }
            
            # Agregar datos del vehículo si están disponibles
            if vehicle_response['status'] == 'success' and vehicle_response['vehicle']:
                vehicle = vehicle_response['vehicle']
                response_data['vehicle_data'] = {
                    'year': vehicle.get('Year', 'N/A'),
                    'type': vehicle.get('Type', 'N/A'),
                    'subtype': vehicle.get('Subtype', 'N/A'),
                    'make': vehicle.get('CarMake', {}).get('CurrentTextValue', 'N/A'),
                    'model': vehicle.get('CarModel', {}).get('CurrentTextValue', 'N/A'),
                    'description': vehicle.get('Description', 'N/A'),
                    'image_url': vehicle.get('ImageUrl', None)
                }
                print(f"✅ Datos del vehículo obtenidos: {vehicle.get('Description', 'N/A')}")
            elif vehicle_response['status'] == 'not_found':
                response_data['message'] = f"Placa {plate_number} procesada, pero no se encontraron datos del vehículo en la base de datos"
            else:
                response_data['message'] = f"Placa {plate_number} procesada, pero hubo un error al consultar los datos del vehículo"
            
            return jsonify(response_data), 200
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar la imagen: {str(e)}'
        }), 500


@bp_plate.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'message': 'Backend de detección de placas funcionando correctamente'
    }), 200
