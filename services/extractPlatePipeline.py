"""
Pipeline integrado de detección y extracción de placas vehiculares
Combina LicensePlateDetector y PlateExtractor en un flujo unificado
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from enum import Enum

import settings
from services.licencePlateDetection import LicensePlateDetector, DetectionStatus, DetectionResult
from services.extractPlate import PlateExtractor, ExtractionStatus, ExtractionResult

class PipelineStatus(Enum):
    """Estados posibles del pipeline completo"""
    SUCCESS = "success"
    NO_PLATE_DETECTED = "no_plate_detected"
    MULTIPLE_PLATES_DETECTED = "multiple_plates_detected"
    NO_TEXT_DETECTED = "no_text_detected"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_FORMAT = "invalid_format"
    ERROR = "error"

class PipelineResult:
    """
    Resultado del pipeline completo de detección y extracción
    
    Attributes:
        status: Estado final del pipeline
        plate_text: Texto de la placa (solo si status == SUCCESS)
        message: Mensaje descriptivo para el usuario
        confidence: Confianza de la extracción OCR
        detection_confidence: Confianza de la detección
        detection_result: Resultado completo de la detección
        extraction_result: Resultado completo de la extracción
    """
    def __init__(
        self,
        status: PipelineStatus,
        message: str,
        plate_text: Optional[str] = None,
        confidence: Optional[float] = None,
        detection_confidence: Optional[float] = None,
        detection_result: Optional[DetectionResult] = None,
        extraction_result: Optional[ExtractionResult] = None
    ):
        self.status = status
        self.message = message
        self.plate_text = plate_text
        self.confidence = confidence
        self.detection_confidence = detection_confidence
        self.detection_result = detection_result
        self.extraction_result = extraction_result
    
    def is_success(self) -> bool:
        """Retorna True si el pipeline fue exitoso"""
        return self.status == PipelineStatus.SUCCESS
    
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario para APIs"""
        result = {
            'status': self.status.value,
            'message': self.message,
            'plate_text': self.plate_text,
            'confidence': self.confidence,
            'detection_confidence': self.detection_confidence
        }
        
        # Agregar detalles adicionales si están disponibles
        if self.detection_result:
            result['detection'] = self.detection_result.to_dict()
        
        if self.extraction_result:
            result['extraction'] = self.extraction_result.to_dict()
        
        return result
    
    def __repr__(self) -> str:
        return f"PipelineResult(status={self.status.value}, plate='{self.plate_text}', conf={self.confidence})"

class PlateRecognitionPipeline:
    """
    Pipeline completo de reconocimiento de placas vehiculares
    
    Integra:
    1. Detección de placa en la imagen (LicensePlateDetector)
    2. Extracción de texto OCR (PlateExtractor)
    3. Validaciones y manejo de errores unificado
    """
    
    def __init__(
        self,
        detector: Optional[LicensePlateDetector] = None,
        extractor: Optional[PlateExtractor] = None
    ):
        """
        Inicializa el pipeline
        
        Args:
            detector: Instancia de LicensePlateDetector (se crea una por defecto)
            extractor: Instancia de PlateExtractor (se crea una por defecto)
        """
        print("=" * 70)
        print("INICIALIZANDO PIPELINE DE RECONOCIMIENTO DE PLACAS")
        print("=" * 70)
        
        # Inicializar detector
        if detector is None:
            print("\n📦 Cargando detector de placas...")
            self.detector = LicensePlateDetector()
            print("✓ Detector cargado")
        else:
            self.detector = detector
        
        # Inicializar extractor OCR
        if extractor is None:
            print("\n📦 Cargando extractor OCR...")
            self.extractor = PlateExtractor(languages=['en', 'es'], gpu=False)
            print("✓ Extractor OCR cargado")
        else:
            self.extractor = extractor
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE INICIALIZADO CORRECTAMENTE")
        print("=" * 70 + "\n")
    
    def process_image(self, image_path: str, save_debug: bool = False) -> PipelineResult:
        """
        Procesa una imagen completa: detecta la placa y extrae el texto
        
        Este es el método principal que debe usarse en producción
        
        Args:
            image_path: Ruta a la imagen del vehículo
            save_debug: Si guardar imágenes intermedias para debug
            
        Returns:
            PipelineResult con el estado y texto de la placa
        """
        try:
            # PASO 1: Detectar placa en la imagen
            detection_result = self.detector.detect_single_plate(str(image_path))
            
            # Manejar errores de detección
            if not detection_result.is_success():
                # Mapear status de detección a status de pipeline
                status_mapping = {
                    DetectionStatus.NO_PLATE_DETECTED: PipelineStatus.NO_PLATE_DETECTED,
                    DetectionStatus.MULTIPLE_PLATES_DETECTED: PipelineStatus.MULTIPLE_PLATES_DETECTED,
                    DetectionStatus.ERROR: PipelineStatus.ERROR
                }
                
                return PipelineResult(
                    status=status_mapping.get(detection_result.status, PipelineStatus.ERROR),
                    message=detection_result.message,
                    detection_confidence=detection_result.confidence,
                    detection_result=detection_result
                )
            
            # Guardar placa detectada si se requiere debug
            if save_debug:
                debug_path = Path(settings.OUTPUT_DIR) / f"debug_{Path(image_path).stem}_plate.jpg"
                cv2.imwrite(str(debug_path), detection_result.plate_image)
            
            # PASO 2: Extraer texto de la placa detectada
            extraction_result = self.extractor.extract_from_image(detection_result.plate_image)
            
            # Manejar errores de extracción
            if not extraction_result.is_success():
                # Mapear status de extracción a status de pipeline
                status_mapping = {
                    ExtractionStatus.NO_TEXT_DETECTED: PipelineStatus.NO_TEXT_DETECTED,
                    ExtractionStatus.LOW_CONFIDENCE: PipelineStatus.LOW_CONFIDENCE,
                    ExtractionStatus.INVALID_FORMAT: PipelineStatus.INVALID_FORMAT,
                    ExtractionStatus.ERROR: PipelineStatus.ERROR
                }
                
                return PipelineResult(
                    status=status_mapping.get(extraction_result.status, PipelineStatus.ERROR),
                    message=extraction_result.message,
                    plate_text=extraction_result.plate_text,
                    confidence=extraction_result.confidence,
                    detection_confidence=detection_result.confidence,
                    detection_result=detection_result,
                    extraction_result=extraction_result
                )
            
            # ÉXITO: Placa detectada y texto extraído correctamente
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                message=f"Placa reconocida: {extraction_result.plate_text}",
                plate_text=extraction_result.plate_text,
                confidence=extraction_result.confidence,
                detection_confidence=detection_result.confidence,
                detection_result=detection_result,
                extraction_result=extraction_result
            )
        
        except Exception as e:
            return PipelineResult(
                status=PipelineStatus.ERROR,
                message=f"Error en el pipeline: {str(e)}"
            )
    
    def process_image_array(self, image: np.ndarray) -> PipelineResult:
        """
        Procesa una imagen en formato array numpy (para APIs que reciben imágenes en memoria)
        
        Args:
            image: Imagen en formato numpy array (BGR)
            
        Returns:
            PipelineResult con el estado y texto de la placa
        """
        try:
            # Guardar temporalmente para procesar
            temp_path = Path(settings.OUTPUT_DIR) / "temp_input.jpg"
            cv2.imwrite(str(temp_path), image)
            
            # Procesar usando el método principal
            result = self.process_image(str(temp_path))
            
            # Limpiar archivo temporal
            if temp_path.exists():
                temp_path.unlink()
            
            return result
        
        except Exception as e:
            return PipelineResult(
                status=PipelineStatus.ERROR,
                message=f"Error al procesar array de imagen: {str(e)}"
            )
    
    def process_batch(self, image_paths: list) -> Dict[str, PipelineResult]:
        """
        Procesa un lote de imágenes
        
        Args:
            image_paths: Lista de rutas a las imágenes
            
        Returns:
            Diccionario {nombre_archivo: PipelineResult}
        """
        results = {}
        total = len(image_paths)
        
        print(f"\n📊 Procesando {total} imágenes en lote...\n")
        
        for idx, image_path in enumerate(image_paths, 1):
            image_name = Path(image_path).name
            print(f"[{idx}/{total}] Procesando {image_name}...")
            
            result = self.process_image(str(image_path))
            results[image_name] = result
            
            if result.is_success():
                print(f"  ✅ {result.plate_text} (conf: {result.confidence:.1%})")
            else:
                print(f"  ⚠️  {result.status.value}")
        
        return results
    
    def get_statistics(self, results: Dict[str, PipelineResult]) -> Dict:
        """
        Calcula estadísticas de un lote procesado
        
        Args:
            results: Diccionario de resultados del proceso batch
            
        Returns:
            Diccionario con estadísticas
        """
        total = len(results)
        success = sum(1 for r in results.values() if r.status == PipelineStatus.SUCCESS)
        no_plate = sum(1 for r in results.values() if r.status == PipelineStatus.NO_PLATE_DETECTED)
        multiple = sum(1 for r in results.values() if r.status == PipelineStatus.MULTIPLE_PLATES_DETECTED)
        no_text = sum(1 for r in results.values() if r.status == PipelineStatus.NO_TEXT_DETECTED)
        low_conf = sum(1 for r in results.values() if r.status == PipelineStatus.LOW_CONFIDENCE)
        invalid = sum(1 for r in results.values() if r.status == PipelineStatus.INVALID_FORMAT)
        errors = sum(1 for r in results.values() if r.status == PipelineStatus.ERROR)
        
        # Confianza promedio de los exitosos
        successful_results = [r for r in results.values() if r.is_success()]
        avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            'total': total,
            'success': success,
            'success_rate': success / total * 100 if total > 0 else 0,
            'no_plate_detected': no_plate,
            'multiple_plates_detected': multiple,
            'no_text_detected': no_text,
            'low_confidence': low_conf,
            'invalid_format': invalid,
            'errors': errors,
            'avg_confidence': avg_confidence
        }