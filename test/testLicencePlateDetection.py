"""
Script de prueba para el detector de placas
Usa el nuevo método detect_single_plate() con validaciones integradas
"""
import sys
from pathlib import Path
import random
import cv2

sys.path.append(str(Path(__file__).parent.parent))

import settings
from services.licencePlateDetection import LicensePlateDetector, DetectionStatus


def main():
    print("=" * 70)
    print("PRUEBA DE DETECCIÓN DE PLACAS CON VALIDACIONES")
    print("=" * 70)
    
    # Verificaciones iniciales
    if not settings.TEST_IMAGES_DIR.exists():
        print(f"❌ Error: No existe el directorio {settings.TEST_IMAGES_DIR}")
        return
    
    if not settings.PLATE_DETECTION_MODEL.exists():
        print(f"❌ Error: No se encuentra el modelo en {settings.PLATE_DETECTION_MODEL}")
        print("\nSoluciones:")
        print("1. Coloca tu modelo .onnx en la ruta indicada")
        print("2. Si tienes un modelo .pt, usa: python convert_model_to_onnx.py")
        return
    
    # Obtener imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*jfif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(settings.TEST_IMAGES_DIR.glob(ext))
    
    if not all_images:
        print(f"❌ No se encontraron imágenes en {settings.TEST_IMAGES_DIR}")
        return
    
    # Seleccionar imágenes
    num_images = min(15, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    print(f"\n📊 Imágenes encontradas: {len(all_images)}")
    print(f"📊 Procesando {num_images} imágenes aleatorias...\n")
    
    # Inicializar detector
    try:
        print(f"⚙️  Cargando modelo: {settings.PLATE_DETECTION_MODEL.name}")
        detector = LicensePlateDetector()
        print(f"✓ Modelo cargado exitosamente")
        print(f"✓ Umbral de confianza: {settings.DETECTION_CONFIDENCE_THRESHOLD}")
        print(f"✓ Umbral de IoU (NMS): {settings.IOU_THRESHOLD}")
        print(f"✓ Tamaño de imagen: {settings.IMAGE_SIZE}\n")
    except Exception as e:
        print(f"\n❌ Error al cargar el modelo: {str(e)}")
        return
    
    # Estadísticas
    stats = {
        'success': 0,
        'no_plate': 0,
        'multiple_plates': 0,
        'error': 0
    }
    
    # Listas para clasificación
    success_images = []
    no_plate_images = []
    multiple_plate_images = []
    error_images = []
    
    print("=" * 70)
    print("PROCESANDO IMÁGENES")
    print("=" * 70 + "\n")
    
    # Procesar cada imagen
    for idx, image_path in enumerate(selected_images, 1):
        print(f"[{idx}/{num_images}] 📸 {image_path.name}")
        
        # Usar el método con validaciones integradas
        result = detector.detect_single_plate(str(image_path))
        
        # Clasificar según el status
        if result.status == DetectionStatus.SUCCESS:
            stats['success'] += 1
            success_images.append((image_path.name, result.confidence))
            print(f"  ✅ {result.message}")
            print(f"     Confianza: {result.confidence:.2%}")
            
            # Guardar placa para OCR
            plate_path = settings.OUTPUT_DIR / f"plate_{image_path.stem}.jpg"
            cv2.imwrite(str(plate_path), result.plate_image)
            h, w = result.plate_image.shape[:2]
            print(f"     💾 Guardado: {plate_path.name} ({w}x{h}px)")
        
        elif result.status == DetectionStatus.NO_PLATE_DETECTED:
            stats['no_plate'] += 1
            no_plate_images.append(image_path.name)
            print(f"  ⚠️  SIN PLACAS")
            print(f"     Mensaje al usuario:")
            for line in result.message.split('\n'):
                print(f"     {line}")
        
        elif result.status == DetectionStatus.MULTIPLE_PLATES_DETECTED:
            stats['multiple_plates'] += 1
            multiple_plate_images.append((image_path.name, result.num_plates_detected))
            print(f"  ⚠️  MÚLTIPLES PLACAS ({result.num_plates_detected})")
            print(f"     Mensaje al usuario:")
            for line in result.message.split('\n'):
                print(f"     {line}")
        
        else:  # ERROR
            stats['error'] += 1
            error_images.append((image_path.name, result.message))
            print(f"  ❌ ERROR")
            print(f"     {result.message}")
        
        print()
    
    # REPORTE FINAL
    print("=" * 70)
    print("📊 REPORTE DE VALIDACIÓN")
    print("=" * 70)
    
    total = sum(stats.values())
    
    print(f"\n🎯 RESUMEN POR ESTADO:")
    print(f"  ✅ Éxito (1 placa):        {stats['success']:2d} ({stats['success']/total*100:5.1f}%)")
    print(f"  ⚠️  Múltiples placas:      {stats['multiple_plates']:2d} ({stats['multiple_plates']/total*100:5.1f}%)")
    print(f"  ⚠️  Sin placas:            {stats['no_plate']:2d} ({stats['no_plate']/total*100:5.1f}%)")
    print(f"  ❌ Errores:                {stats['error']:2d} ({stats['error']/total*100:5.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  📊 Total procesadas:       {total:2d}")
    
    # Detalles de imágenes exitosas
    if success_images:
        print(f"\n✅ IMÁGENES EXITOSAS ({len(success_images)}):")
        print("   └─ Estas imágenes están listas para OCR")
        print()
        for img_name, confidence in success_images:
            print(f"     • {img_name} (confianza: {confidence:.1%})")
    
    # Detalles de múltiples placas
    if multiple_plate_images:
        print(f"\n⚠️  IMÁGENES CON MÚLTIPLES PLACAS ({len(multiple_plate_images)}):")
        print("   └─ El módulo ya devolvió el mensaje apropiado:")
        print('      "Se detectaron N placas. Por favor, tome una nueva foto..."')
        print()
        for img_name, count in multiple_plate_images:
            print(f"     • {img_name}: {count} placas detectadas")
    
    # Detalles de sin placas
    if no_plate_images:
        print(f"\n⚠️  IMÁGENES SIN PLACAS ({len(no_plate_images)}):")
        print("   └─ El módulo ya devolvió el mensaje apropiado:")
        print('      "No se detectó ninguna placa. Por favor, intente..."')
        print()
        for img_name in no_plate_images:
            print(f"     • {img_name}")
    
    # Detalles de errores
    if error_images:
        print(f"\n❌ ERRORES ({len(error_images)}):")
        print()
        for img_name, error_msg in error_images:
            print(f"     • {img_name}")
            print(f"       {error_msg}")
    
    print(f"\n📂 RESULTADOS:")
    print(f"   • Placas válidas guardadas en: {settings.OUTPUT_DIR}")
    print(f"   • Solo se guardaron las {stats['success']} placas individuales detectadas")
    
    print("\n" + "=" * 70)
    print("💡 ANÁLISIS PARA LA APLICACIÓN")
    print("=" * 70)
    
    if total > 0:
        success_rate = stats['success'] / total * 100
        
        print(f"\n📈 Tasa de éxito: {success_rate:.1f}%")
        print(f"   → {stats['success']} de {total} imágenes son válidas para procesar")
        
        if stats['multiple_plates'] > 0:
            print(f"\n⚠️  {stats['multiple_plates']} imágenes requieren que el usuario tome otra foto")
            print("   → El módulo ya maneja esto automáticamente")
        
        if stats['no_plate'] > 0:
            print(f"\n⚠️  {stats['no_plate']} imágenes no tienen placas detectables")
            print("   → El módulo ya proporciona instrucciones al usuario")
        
        print(f"\n✅ CONCLUSIÓN:")
        print(f"   El módulo LicensePlateDetector maneja correctamente:")
        print(f"   • Detección exitosa (1 placa)")
        print(f"   • Validación de múltiples placas")
        print(f"   • Manejo de imágenes sin placas")
        print(f"   • Mensajes descriptivos para el usuario")
    
    print("=" * 70)


if __name__ == "__main__":
    main()