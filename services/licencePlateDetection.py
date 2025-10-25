"""
Servicio de detección de placas vehiculares usando ONNX Runtime
Incluye validaciones de casos de uso para aplicaciones móviles
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from enum import Enum
import settings


class DetectionStatus(Enum):
    """Estados posibles de la detección"""
    SUCCESS = "success"
    NO_PLATE_DETECTED = "no_plate_detected"
    MULTIPLE_PLATES_DETECTED = "multiple_plates_detected"
    ERROR = "error"


class DetectionResult:
    """
    Resultado de la detección de placas
    
    Attributes:
        status: Estado de la detección
        plate_image: Imagen de la placa (solo si status == SUCCESS)
        message: Mensaje descriptivo para el usuario
        num_plates_detected: Número de placas detectadas
        all_plates: Lista de todas las placas detectadas (para debug)
        confidence: Confianza de la detección (solo si status == SUCCESS)
    """
    def __init__(
        self,
        status: DetectionStatus,
        message: str,
        plate_image: Optional[np.ndarray] = None,
        num_plates_detected: int = 0,
        all_plates: Optional[List[np.ndarray]] = None,
        confidence: Optional[float] = None
    ):
        self.status = status
        self.message = message
        self.plate_image = plate_image
        self.num_plates_detected = num_plates_detected
        self.all_plates = all_plates or []
        self.confidence = confidence
    
    def is_success(self) -> bool:
        """Retorna True si la detección fue exitosa"""
        return self.status == DetectionStatus.SUCCESS
    
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario para APIs"""
        return {
            'status': self.status.value,
            'message': self.message,
            'num_plates_detected': self.num_plates_detected,
            'confidence': self.confidence,
            'has_plate_image': self.plate_image is not None
        }
    
    def __repr__(self) -> str:
        return f"DetectionResult(status={self.status.value}, num_plates={self.num_plates_detected})"


class LicensePlateDetector:
    """Detector de placas vehiculares usando modelo YOLO en formato ONNX"""
    
    def __init__(self, model_path: Optional[Path] = None, iou_threshold: Optional[float] = None):
        """
        Inicializa el detector de placas
        
        Args:
            model_path: Ruta al modelo ONNX. Si es None, usa la ruta de settings
            iou_threshold: Umbral de IoU para NMS. Si es None, usa el de settings
        """
        self.model_path = model_path or settings.PLATE_DETECTION_MODEL
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.image_size = settings.IMAGE_SIZE
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD
        
        # Cargar modelo ONNX con configuración flexible
        try:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 4
            
            providers = ['CPUExecutionProvider']
            provider_options = [{}]
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
        except Exception as e:
            error_msg = str(e)
            
            if "opset" in error_msg.lower():
                print("⚠️  Advertencia: Modelo con opset nuevo, intentando cargar de todas formas...")
                try:
                    import warnings
                    warnings.filterwarnings('ignore')
                    
                    self.session = ort.InferenceSession(
                        str(self.model_path),
                        providers=['CPUExecutionProvider']
                    )
                    
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_names = [output.name for output in self.session.get_outputs()]
                    print("✓ Modelo cargado exitosamente (modo compatible)")
                    
                except:
                    raise RuntimeError(
                        f"Tu modelo usa opset 22 que es muy nuevo.\n"
                        f"Solución: Instala onnxruntime-gpu o una versión más reciente:\n"
                        f"  pip install --upgrade onnxruntime\n"
                        f"O re-exporta tu modelo .pt con opset 12:\n"
                        f"  python convert_model_to_onnx.py"
                    )
            else:
                raise RuntimeError(f"Error al cargar el modelo ONNX: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocesa la imagen para el modelo YOLO
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            Tupla con (imagen procesada, ratio de escala, padding)
        """
        height, width = image.shape[:2]
        
        ratio = self.image_size / max(height, width)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        padded = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        
        pad_x = (self.image_size - new_width) // 2
        pad_y = (self.image_size - new_height) // 2
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, ratio, (pad_x, pad_y)
    
    def compute_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calcula el IoU (Intersection over Union) entre dos cajas
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU score entre 0 y 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_nms(self, detections: List[Tuple[int, int, int, int, float]], iou_threshold: float = 0.45) -> List[Tuple[int, int, int, int, float]]:
        """
        Aplica Non-Maximum Suppression para eliminar detecciones duplicadas
        
        Args:
            detections: Lista de detecciones [(x1, y1, x2, y2, confidence), ...]
            iou_threshold: Umbral de IoU para considerar cajas duplicadas
            
        Returns:
            Lista filtrada de detecciones
        """
        if not detections:
            return []
        
        # Ordenar por confianza (mayor a menor)
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        
        while detections:
            # Tomar la detección con mayor confianza
            best = detections.pop(0)
            keep.append(best)
            
            # Filtrar detecciones que se solapan demasiado con la mejor
            filtered = []
            for det in detections:
                iou = self.compute_iou(best[:4], det[:4])
                if iou < iou_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def postprocess_detections(
        self, 
        outputs: np.ndarray, 
        ratio: float, 
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Procesa las salidas del modelo y devuelve las detecciones
        Incluye NMS para eliminar detecciones duplicadas
        
        Args:
            outputs: Salida del modelo ONNX
            ratio: Ratio de escala usado en preprocesamiento
            padding: Padding aplicado (pad_x, pad_y)
            original_shape: Forma original de la imagen (height, width)
            
        Returns:
            Lista de detecciones [(x1, y1, x2, y2, confidence), ...]
        """
        pad_x, pad_y = padding
        orig_height, orig_width = original_shape
        
        predictions = outputs
        
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        detections = []
        
        for pred in predictions:
            if len(pred) < 5:
                continue
            
            x_center, y_center, w, h, confidence = pred[:5]
            
            if confidence < self.confidence_threshold:
                continue
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            x1 = (x1 - pad_x) / ratio
            y1 = (y1 - pad_y) / ratio
            x2 = (x2 - pad_x) / ratio
            y2 = (y2 - pad_y) / ratio
            
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            if x2 > x1 and y2 > y1:
                detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        # Aplicar NMS para eliminar detecciones duplicadas
        detections = self.apply_nms(detections, iou_threshold=self.iou_threshold)
        
        return detections
    
    def detect_single_plate(self, image_path: str) -> DetectionResult:
        """
        Detecta UNA SOLA placa en la imagen (caso de uso principal)
        
        Este método implementa la lógica de validación para aplicaciones móviles:
        - Si detecta 0 placas: retorna error indicando que no se encontró placa
        - Si detecta 1 placa: retorna éxito con la imagen de la placa
        - Si detecta >1 placas: retorna error indicando que hay múltiples placas
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            DetectionResult con el estado y mensaje apropiado
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return DetectionResult(
                    status=DetectionStatus.ERROR,
                    message=f"No se pudo leer la imagen: {image_path}",
                    num_plates_detected=0
                )
            
            orig_height, orig_width = image.shape[:2]
            
            # Preprocesar
            input_tensor, ratio, padding = self.preprocess_image(image)
            
            # Inferencia
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Postprocesar
            detections = self.postprocess_detections(
                outputs[0], 
                ratio, 
                padding, 
                (orig_height, orig_width)
            )
            
            # Extraer todas las placas
            all_plates = []
            confidences = []
            for x1, y1, x2, y2, conf in detections:
                plate = image[y1:y2, x1:x2]
                if plate.size > 0:
                    all_plates.append(plate)
                    confidences.append(conf)
            
            num_plates = len(all_plates)
            
            # VALIDACIÓN: Sin placas detectadas
            if num_plates == 0:
                return DetectionResult(
                    status=DetectionStatus.NO_PLATE_DETECTED,
                    message="No se detectó ninguna placa. Por favor, intente:\n"
                            "• Acercarse más al vehículo\n"
                            "• Asegurar buena iluminación\n"
                            "• Centrar la placa en la imagen",
                    num_plates_detected=0
                )
            
            # VALIDACIÓN: Múltiples placas detectadas
            elif num_plates > 1:
                return DetectionResult(
                    status=DetectionStatus.MULTIPLE_PLATES_DETECTED,
                    message=f"Se detectaron {num_plates} placas en la imagen.\n"
                            f"Por favor, tome una nueva foto donde solo aparezca "
                            f"la placa que desea consultar.\n\n"
                            f"Consejo: Acérquese al vehículo y centre la placa.",
                    num_plates_detected=num_plates,
                    all_plates=all_plates
                )
            
            # ÉXITO: Una sola placa detectada
            else:
                return DetectionResult(
                    status=DetectionStatus.SUCCESS,
                    message="Placa detectada correctamente",
                    plate_image=all_plates[0],
                    num_plates_detected=1,
                    all_plates=all_plates,
                    confidence=confidences[0]
                )
        
        except Exception as e:
            return DetectionResult(
                status=DetectionStatus.ERROR,
                message=f"Error durante la detección: {str(e)}",
                num_plates_detected=0
            )
    
    def detect(self, image_path: str) -> List[np.ndarray]:
        """
        Detecta y extrae TODAS las placas de una imagen (método legacy)
        
        NOTA: Para uso en aplicaciones móviles, usar detect_single_plate() en su lugar
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Lista de imágenes recortadas con las placas detectadas
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")
        
        orig_height, orig_width = image.shape[:2]
        
        input_tensor, ratio, padding = self.preprocess_image(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess_detections(
            outputs[0], 
            ratio, 
            padding, 
            (orig_height, orig_width)
        )
        
        plates = []
        for x1, y1, x2, y2, conf in detections:
            plate = image[y1:y2, x1:x2]
            if plate.size > 0:
                plates.append(plate)
        
        return plates
    
    def detect_with_boxes(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        Detecta placas y devuelve la imagen con las coordenadas de las cajas
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tupla con (imagen original, lista de detecciones)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")
        
        orig_height, orig_width = image.shape[:2]
        
        input_tensor, ratio, padding = self.preprocess_image(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess_detections(
            outputs[0], 
            ratio, 
            padding, 
            (orig_height, orig_width)
        )
        
        return image, detections