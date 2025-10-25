"""
Servicio MEJORADO de extracción de texto de placas vehiculares usando EasyOCR
Incorpora mejores prácticas investigadas:
- Super-resolución mediante interpolación inteligente
- Múltiples técnicas de preprocesamiento
- Allowlist para restringir caracteres
- Siempre retorna mejor candidato (incluso con baja confianza)
- Correcciones de caracteres comunes más robustas
- Eliminación de guiones al final del procesamiento
"""
import cv2
import numpy as np
import easyocr
import re
import string
from typing import Optional, List, Tuple, Dict
from enum import Enum

class ExtractionStatus(Enum):
    """Estados posibles de la extracción OCR"""
    SUCCESS = "success"
    NO_TEXT_DETECTED = "no_text_detected"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_FORMAT = "invalid_format"
    ERROR = "error"

class ExtractionResult:
    """
    Resultado de la extracción de texto de placa
    
    Attributes:
        status: Estado de la extracción
        plate_text: Texto de la placa limpio (SIEMPRE presente si hay detección)
        raw_text: Texto crudo antes de limpieza
        confidence: Confianza promedio de la extracción
        message: Mensaje descriptivo
        all_detections: Todas las detecciones OCR (para debug)
    """
    def __init__(
        self,
        status: ExtractionStatus,
        message: str,
        plate_text: Optional[str] = None,
        raw_text: Optional[str] = None,
        confidence: Optional[float] = None,
        all_detections: Optional[List[Tuple[str, float]]] = None
    ):
        self.status = status
        self.message = message
        self.plate_text = plate_text
        self.raw_text = raw_text
        self.confidence = confidence
        self.all_detections = all_detections or []
    
    def is_success(self) -> bool:
        """Retorna True si la extracción fue exitosa"""
        return self.status == ExtractionStatus.SUCCESS
    
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario para APIs"""
        return {
            'status': self.status.value,
            'message': self.message,
            'plate_text': self.plate_text,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'num_detections': len(self.all_detections)
        }
    
    def __repr__(self) -> str:
        return f"ExtractionResult(status={self.status.value}, plate='{self.plate_text}', conf={self.confidence})"

class PlateExtractor:
    """
    Extractor MEJORADO de texto de placas vehiculares usando EasyOCR
    Implementa super-resolución, múltiples preprocesamientos y allowlist
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Inicializa el extractor OCR
        
        Args:
            languages: Lista de idiomas para OCR (default: ['en'])
            gpu: Si usar GPU para aceleración (default: False)
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        
        # Umbrales más permisivos - SIEMPRE retornamos el mejor candidato
        self.confidence_threshold_warning = 0.6  # Solo para advertencia
        self.confidence_threshold_min = 0.0      # Sin rechazo por confianza
        
        # Inicializar EasyOCR
        print(f"🔧 Inicializando EasyOCR con idiomas: {self.languages}")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        print("✓ EasyOCR inicializado correctamente")
        
        # ALLOWLIST: Solo letras mayúsculas, números y guión
        # Esto evita que OCR detecte símbolos extraños
        self.allowed_chars = string.ascii_uppercase + string.digits + '-'
        
        # Patrones de placas vehiculares comunes (más flexibles)
        self.plate_patterns = [
            r'^[A-Z]{3}-?\d{3,4}$',       # ABC-123 o ABC123 o ABC-1234 o ABC1234
            r'^[A-Z]{2,4}\d{3,5}$',       # AB123, ABC1234, ABCD12345
            r'^[A-Z]{2}-?\d{3,4}$',       # AB-123 o AB123
            r'^\d{3,4}[A-Z]{2,3}$',       # 123ABC, 1234ABC
            r'^\d{3,4}-?[A-Z]{2,3}$',     # 123-ABC o 123ABC
            r'^[A-Z]{1,3}\d{2,4}[A-Z]{0,2}$',  # Formato mixto
            r'^\d{3,4}$',                 # Solo números (placas viejas)
        ]
    
    def upscale_image(self, image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Aumenta resolución de imagen usando interpolación bicúbica
        Investigación: Super-resolution mejora OCR en placas pequeñas
        """
        if image.size == 0:
            return image
        
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Bicubic es mejor que bilinear para texto
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Mejora contraste local sin sobre-amplificar ruido
        """
        if len(image.shape) == 3:
            # Convertir a LAB para aplicar CLAHE solo en luminancia
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce ruido manteniendo bordes
        fastNlMeansDenoisingColored es más efectivo que GaussianBlur
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aumenta nitidez para mejorar reconocimiento de bordes de caracteres
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def increase_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """Aumenta el contraste de la imagen"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def preprocess_plate_image_basic(self, image: np.ndarray) -> np.ndarray:
        """Preprocesamiento básico (actual)"""
        return self.increase_contrast(image, alpha=1.5, beta=0)
    
    def preprocess_plate_image_aggressive(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento agresivo combinando múltiples técnicas
        Según investigación: CLAHE + denoising + sharpening mejora OCR
        """
        # 1. Super-resolución (x2)
        upscaled = self.upscale_image(image, scale_factor=2.0)
        
        # 2. Reducir ruido
        denoised = self.denoise_image(upscaled)
        
        # 3. CLAHE para mejorar contraste local
        enhanced = self.adaptive_histogram_equalization(denoised)
        
        # 4. Aumentar nitidez
        sharpened = self.sharpen_image(enhanced)
        
        # 5. Contraste final
        final = self.increase_contrast(sharpened, alpha=1.3, beta=10)
        
        return final
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído con correcciones OCR más inteligentes
        """
        if not text:
            return ""
        
        text = text.upper()
        
        # Solo mantener caracteres permitidos
        text = ''.join(c for c in text if c in self.allowed_chars)
        
        # CORRECCIONES INTELIGENTES basadas en contexto
        # Regla: Letras confundidas con números en POSICIONES DE LETRA
        # Regla: Números confundidos con letras en POSICIONES DE NÚMERO
        
        # Para placas ABC-123 o ABC123:
        # Primeros 2-3 caracteres suelen ser LETRAS
        # Últimos 3-4 caracteres suelen ser NÚMEROS
        
        result = list(text)
        text_len = len(result)
        
        # Detectar separador (guión)
        dash_pos = text.find('-')
        
        if dash_pos != -1:
            # Formato ABC-123
            # Antes del guión: solo letras
            for i in range(dash_pos):
                if result[i].isdigit():
                    # Número -> Letra
                    if result[i] == '0': result[i] = 'O'
                    elif result[i] == '1': result[i] = 'I'
                    elif result[i] == '5': result[i] = 'S'
                    elif result[i] == '8': result[i] = 'B'
                    elif result[i] == '6': result[i] = 'G'
            
            # Después del guión: solo números
            for i in range(dash_pos + 1, text_len):
                if result[i].isalpha():
                    # Letra -> Número
                    if result[i] == 'O': result[i] = '0'
                    elif result[i] == 'I' or result[i] == 'L': result[i] = '1'
                    elif result[i] == 'S': result[i] = '5'
                    elif result[i] == 'B': result[i] = '8'
                    elif result[i] == 'Z': result[i] = '2'
                    elif result[i] == 'G': result[i] = '6'
                    elif result[i] == 'T': result[i] = '7'
        else:
            # Sin guión: heurística basada en posición
            # Primeros 40% = letras, últimos 60% = números
            letter_end = int(text_len * 0.4)
            
            for i in range(letter_end):
                if result[i].isdigit():
                    if result[i] == '0': result[i] = 'O'
                    elif result[i] == '1': result[i] = 'I'
                    elif result[i] == '5': result[i] = 'S'
                    elif result[i] == '8': result[i] = 'B'
            
            for i in range(letter_end, text_len):
                if result[i].isalpha():
                    if result[i] == 'O': result[i] = '0'
                    elif result[i] == 'I' or result[i] == 'L': result[i] = '1'
                    elif result[i] == 'S': result[i] = '5'
                    elif result[i] == 'B': result[i] = '8'
                    elif result[i] == 'Z': result[i] = '2'
        
        return ''.join(result)
    
    def remove_hyphens(self, text: str) -> str:
        """
        Elimina todos los guiones del texto
        Se aplica al final del procesamiento
        """
        return text.replace('-', '')
    
    def get_bbox_center(self, bbox: List) -> Tuple[float, float]:
        """Obtiene las coordenadas del centro de un bbox"""
        points = np.array(bbox)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        return center_x, center_y
    
    def get_bbox_bounds(self, bbox: List) -> Tuple[float, float, float, float]:
        """Obtiene los límites de un bbox"""
        points = np.array(bbox)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    
    def should_group_detections(self, bbox1: List, bbox2: List, image_height: int) -> bool:
        """Determina si dos detecciones deben agruparse"""
        x1_min, x1_max, y1_min, y1_max = self.get_bbox_bounds(bbox1)
        x2_min, x2_max, y2_min, y2_max = self.get_bbox_bounds(bbox2)
        
        height1 = y1_max - y1_min
        height2 = y2_max - y2_min
        center1_y = (y1_min + y1_max) / 2
        center2_y = (y2_min + y2_max) / 2
        
        vertical_distance = abs(center2_y - center1_y)
        avg_height = (height1 + height2) / 2
        
        if vertical_distance > avg_height * 1.5:
            return False
        
        horizontal_gap = max(0, min(x2_min - x1_max, x1_min - x2_max))
        avg_width = ((x1_max - x1_min) + (x2_max - x2_min)) / 2
        
        if horizontal_gap > avg_width * 0.5:
            return False
        
        return True
    
    def group_detections_spatially(self, detections: List[Dict], image_height: int) -> List[List[Dict]]:
        """Agrupa detecciones espacialmente cercanas"""
        if not detections:
            return []
        
        sorted_detections = sorted(detections, key=lambda d: (
            self.get_bbox_center(d['bbox'])[1],
            self.get_bbox_center(d['bbox'])[0]
        ))
        
        groups = []
        current_group = [sorted_detections[0]]
        
        for i in range(1, len(sorted_detections)):
            current_det = sorted_detections[i]
            prev_det = current_group[-1]
            
            if self.should_group_detections(prev_det['bbox'], current_det['bbox'], image_height):
                current_group.append(current_det)
            else:
                groups.append(current_group)
                current_group = [current_det]
        
        groups.append(current_group)
        return groups
    
    def score_plate_candidate(self, text: str, confidence: float, num_parts: int) -> float:
        """
        Calcula score para un candidato de placa
        MEJORADO: Mejor balance entre confianza y formato
        """
        score = confidence
        
        # LONGITUD
        if 5 <= len(text) <= 9:
            score *= 1.8
        elif 4 <= len(text) <= 10:
            score *= 1.3
        elif len(text) == 3:
            score *= 0.7  # Placas muy cortas son posibles
        elif len(text) < 3:
            score *= 0.3
        else:
            score *= 0.4
        
        # BALANCE LETRAS/NÚMEROS
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        num_digits = sum(c.isdigit() for c in text)
        num_letters = sum(c.isalpha() for c in text)
        
        if has_letters and has_digits:
            score *= 2.5  # Alfanumérico = MÁXIMA PRIORIDAD
            
            if 1 <= num_letters <= 4 and 2 <= num_digits <= 5:
                score *= 1.4
        elif has_digits and not has_letters:
            # Solo números: BAJA prioridad pero posible (placas antiguas)
            if len(text) == 3 or len(text) == 4:
                score *= 0.6  # Más tolerante con placas numéricas cortas
            else:
                score *= 0.3
        else:
            score *= 0.1
        
        # PATRONES
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                score *= 1.6
                break
        
        # GUIÓN presente
        if '-' in text:
            score *= 1.2
        
        # AGRUPACIÓN (texto fragmentado reconstruido)
        if num_parts > 1:
            score *= 1.2
        
        return score
    
    def validate_plate_format(self, text: str) -> bool:
        """Validación MÁS FLEXIBLE"""
        if not text or len(text) < 3:
            return False
        
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        
        # Criterio flexible: debe tener letras O números
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        
        # Acepta solo números si son 3-4 dígitos (placas antiguas)
        if has_digits and not has_letters and 3 <= len(text) <= 4:
            return True
        
        # Acepta alfanumérico de 4-10 caracteres
        return (has_letters or has_digits) and 3 <= len(text) <= 10
    
    def run_ocr_with_config(self, image: np.ndarray, config_name: str) -> List:
        """
        Ejecuta OCR con configuración específica
        """
        if config_name == 'basic':
            return self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                width_ths=0.5,
                ycenter_ths=0.3,
                allowlist=self.allowed_chars,  # IMPORTANTE
                contrast_ths=0.1,
                adjust_contrast=0.5
            )
        elif config_name == 'aggressive':
            return self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                width_ths=0.3,  # Más sensible a texto separado
                ycenter_ths=0.5,
                allowlist=self.allowed_chars,
                contrast_ths=0.05,  # Más pases con ajuste de contraste
                adjust_contrast=0.7,
                text_threshold=0.6,  # Menos estricto
                link_threshold=0.3
            )
        else:
            return []
    
    def extract_from_image(self, image: np.ndarray) -> ExtractionResult:
        """
        Extrae el texto de la placa desde una imagen
        MEJORADO: Múltiples estrategias, siempre retorna mejor candidato
        """
        try:
            if image is None or image.size == 0:
                return ExtractionResult(
                    status=ExtractionStatus.ERROR,
                    message="Imagen inválida o vacía"
                )
            
            image_height = image.shape[0]
            
            # ESTRATEGIA MULTI-PASS: Probar diferentes preprocesamientos
            all_candidates = []
            
            # PASS 1: Preprocesamiento básico
            print("   🔍 Pass 1: Preprocesamiento básico")
            preprocessed_basic = self.preprocess_plate_image_basic(image)
            results_basic = self.run_ocr_with_config(preprocessed_basic, 'basic')
            
            if results_basic:
                candidates_basic = self._process_ocr_results(results_basic, image_height, "basic")
                all_candidates.extend(candidates_basic)
                print(f"      → {len(candidates_basic)} candidatos detectados")
            
            # PASS 2: Preprocesamiento agresivo (si Pass 1 no fue muy exitoso)
            if not results_basic or (results_basic and max([r[2] for r in results_basic], default=0) < 0.7):
                print("   🔍 Pass 2: Preprocesamiento agresivo (super-res + CLAHE)")
                preprocessed_aggressive = self.preprocess_plate_image_aggressive(image)
                results_aggressive = self.run_ocr_with_config(preprocessed_aggressive, 'aggressive')
                
                if results_aggressive:
                    candidates_aggressive = self._process_ocr_results(results_aggressive, image_height, "aggressive")
                    all_candidates.extend(candidates_aggressive)
                    print(f"      → {len(candidates_aggressive)} candidatos adicionales")
            
            # Verificar si hay detecciones
            if not all_candidates:
                return ExtractionResult(
                    status=ExtractionStatus.NO_TEXT_DETECTED,
                    message="No se detectó texto en la placa.\n"
                            "Sugerencias:\n"
                            "• Asegurar buena iluminación\n"
                            "• Enfocar la placa\n"
                            "• Limpiar la placa si está sucia",
                    plate_text="",  # Retorna string vacío si no hay detección
                    raw_text="",
                    confidence=0.0
                )
            
            # Ordenar TODOS los candidatos por score
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Debug: Top 5
            print(f"   🏆 Top 5 candidatos globales:")
            for i, cand in enumerate(all_candidates[:5], 1):
                parts = f" ({cand['num_parts']}p)" if cand['num_parts'] > 1 else ""
                source = f" [{cand['source']}]"
                print(f"      {i}. '{cand['text']}' → score:{cand['score']:.2f} conf:{cand['confidence']:.1%}{parts}{source}")
            
            # SIEMPRE tomar el mejor candidato (sin importar confianza)
            best = all_candidates[0]
            best_text = best['text']
            best_confidence = best['confidence']
            
            # ELIMINAR GUIONES AL FINAL
            final_plate_text = self.remove_hyphens(best_text)
            
            # Determinar STATUS basado en confianza y validación
            # PERO SIEMPRE retornar plate_text
            
            if best_confidence >= self.confidence_threshold_warning:
                # Alta confianza
                if self.validate_plate_format(best_text):
                    return ExtractionResult(
                        status=ExtractionStatus.SUCCESS,
                        message="Placa extraída correctamente",
                        plate_text=final_plate_text,
                        raw_text=best_text,
                        confidence=best_confidence,
                        all_detections=[(self.remove_hyphens(c['text']), c['confidence']) for c in all_candidates[:5]]
                    )
                else:
                    # Formato inválido pero buena confianza
                    return ExtractionResult(
                        status=ExtractionStatus.INVALID_FORMAT,
                        message=f"Texto detectado '{final_plate_text}' con buena confianza pero formato inusual.\n"
                                f"Confianza: {best_confidence:.1%}.\n"
                                f"⚠️ Verificar manualmente si es correcto.",
                        plate_text=final_plate_text,
                        raw_text=best_text,
                        confidence=best_confidence,
                        all_detections=[(self.remove_hyphens(c['text']), c['confidence']) for c in all_candidates[:5]]
                    )
            else:
                # Baja confianza PERO retornamos el texto de todas formas
                if self.validate_plate_format(best_text):
                    return ExtractionResult(
                        status=ExtractionStatus.LOW_CONFIDENCE,
                        message=f"Placa detectada: '{final_plate_text}' (confianza: {best_confidence:.1%}).\n"
                                f"⚠️ Confianza baja. Considerar:\n"
                                f"• Mejorar iluminación\n"
                                f"• Tomar foto más cerca\n"
                                f"• Verificar resultado manualmente",
                        plate_text=final_plate_text,
                        raw_text=best_text,
                        confidence=best_confidence,
                        all_detections=[(self.remove_hyphens(c['text']), c['confidence']) for c in all_candidates[:5]]
                    )
                else:
                    return ExtractionResult(
                        status=ExtractionStatus.INVALID_FORMAT,
                        message=f"Texto detectado: '{final_plate_text}' (confianza: {best_confidence:.1%}).\n"
                                f"⚠️ No coincide con formato de placa típico.\n"
                                f"Verificar que solo la placa esté en la imagen.",
                        plate_text=final_plate_text,
                        raw_text=best_text,
                        confidence=best_confidence,
                        all_detections=[(self.remove_hyphens(c['text']), c['confidence']) for c in all_candidates[:5]]
                    )
        
        except Exception as e:
            return ExtractionResult(
                status=ExtractionStatus.ERROR,
                message=f"Error durante la extracción OCR: {str(e)}",
                plate_text="",  # Retorna string vacío en caso de error
                raw_text="",
                confidence=0.0
            )
    
    def _process_ocr_results(self, results: List, image_height: int, source: str) -> List[Dict]:
        """
        Procesa resultados de OCR y genera candidatos
        """
        detections = []
        
        for (bbox, text, confidence) in results:
            cleaned_text = self.clean_text(text)
            if cleaned_text and len(cleaned_text) >= 1:  # Más permisivo
                detections.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        if not detections:
            return []
        
        # Agrupar espacialmente
        groups = self.group_detections_spatially(detections, image_height)
        
        candidates = []
        
        # Candidatos individuales
        for det in detections:
            score = self.score_plate_candidate(det['text'], det['confidence'], 1)
            candidates.append({
                'text': det['text'],
                'confidence': det['confidence'],
                'score': score,
                'num_parts': 1,
                'source': source
            })
        
        # Candidatos agrupados
        for group in groups:
            if len(group) > 1:
                group_sorted = sorted(group, key=lambda d: (
                    self.get_bbox_center(d['bbox'])[1],
                    self.get_bbox_center(d['bbox'])[0]
                ))
                
                combined_text = ''.join(d['text'] for d in group_sorted)
                avg_confidence = sum(d['confidence'] for d in group_sorted) / len(group_sorted)
                
                score = self.score_plate_candidate(combined_text, avg_confidence, len(group))
                
                candidates.append({
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'score': score,
                    'num_parts': len(group),
                    'source': source
                })
        
        return candidates
    
    def extract_from_path(self, image_path: str) -> ExtractionResult:
        """Extrae el texto desde una ruta de imagen"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return ExtractionResult(
                    status=ExtractionStatus.ERROR,
                    message=f"No se pudo leer la imagen: {image_path}",
                    plate_text="",
                    raw_text="",
                    confidence=0.0
                )
            
            return self.extract_from_image(image)
        
        except Exception as e:
            return ExtractionResult(
                status=ExtractionStatus.ERROR,
                message=f"Error al leer la imagen: {str(e)}",
                plate_text="",
                raw_text="",
                confidence=0.0
            )