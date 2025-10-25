"""
Script de prueba para el pipeline completo de reconocimiento de placas
Testa la integración de detección + OCR
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

import settings
from services.extractPlatePipeline import PlateRecognitionPipeline, PipelineStatus

def print_separator(char="=", length=70):
    """Imprime una línea separadora"""
    print(char * length)

def print_result_detail(image_name: str, result, duration: float):
    """Imprime los detalles de un resultado"""
    print(f"\n📸 Imagen: {image_name}")
    print(f"⏱️  Tiempo: {duration:.2f}s")
    
    if result.is_success():
        print(f"✅ Estado: {result.status.value.upper()}")
        print(f"🔢 Placa: {result.plate_text}")
        print(f"📊 Confianza OCR: {result.confidence:.1%}")
        print(f"📊 Confianza Detección: {result.detection_confidence:.1%}")
        print(f"💬 Mensaje: {result.message}")
    else:
        # Mostrar error con su categoría
        status_emoji = {
            PipelineStatus.NO_PLATE_DETECTED: "🚫",
            PipelineStatus.MULTIPLE_PLATES_DETECTED: "⚠️",
            PipelineStatus.NO_TEXT_DETECTED: "📝",
            PipelineStatus.LOW_CONFIDENCE: "📉",
            PipelineStatus.INVALID_FORMAT: "❌",
            PipelineStatus.ERROR: "💥"
        }
        
        emoji = status_emoji.get(result.status, "⚠️")
        print(f"{emoji} Estado: {result.status.value.upper()}")
        print(f"💬 Mensaje para usuario:")
        for line in result.message.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Mostrar detalles adicionales si están disponibles
        if result.plate_text:
            print(f"🔍 Texto detectado (no validado): {result.plate_text}")
        if result.confidence:
            print(f"📊 Confianza: {result.confidence:.1%}")

def main():
    print_separator("=")
    print("PRUEBA DEL PIPELINE COMPLETO DE RECONOCIMIENTO DE PLACAS")
    print_separator("=")
    
    # Verificaciones iniciales
    if not settings.TEST_IMAGES_DIR.exists():
        print(f"\n❌ Error: No existe el directorio {settings.TEST_IMAGES_DIR}")
        print("\nCrea el directorio y coloca imágenes de prueba:")
        print(f"   mkdir -p {settings.TEST_IMAGES_DIR}")
        return
    
    if not settings.PLATE_DETECTION_MODEL.exists():
        print(f"\n❌ Error: No se encuentra el modelo en {settings.PLATE_DETECTION_MODEL}")
        return
    
    # Obtener todas las imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.jfif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(settings.TEST_IMAGES_DIR.glob(ext))
    
    if not all_images:
        print(f"\n❌ No se encontraron imágenes en {settings.TEST_IMAGES_DIR}")
        print("\nColoca imágenes de prueba en ese directorio")
        return
    
    print(f"\n📊 Imágenes encontradas: {len(all_images)}")
    print(f"📁 Directorio: {settings.TEST_IMAGES_DIR}")
    
    # Inicializar pipeline
    print("\n")
    print_separator("-")
    try:
        pipeline = PlateRecognitionPipeline()
    except Exception as e:
        print(f"\n❌ Error al inicializar el pipeline: {str(e)}")
        return
    
    # Procesar todas las imágenes
    print_separator("=")
    print("PROCESANDO IMÁGENES")
    print_separator("=")
    
    results = {}
    total_time = 0
    
    for idx, image_path in enumerate(sorted(all_images), 1):
        print(f"\n[{idx}/{len(all_images)}] {'─' * 60}")
        
        start_time = time.time()
        result = pipeline.process_image(str(image_path))
        duration = time.time() - start_time
        total_time += duration
        
        image_name = image_path.name
        results[image_name] = result
        
        print_result_detail(image_name, result, duration)
    
    # REPORTE FINAL
    print("\n")
    print_separator("=")
    print("📊 REPORTE FINAL DEL PIPELINE")
    print_separator("=")
    
    stats = pipeline.get_statistics(results)
    
    print(f"\n🎯 RESUMEN GENERAL:")
    print(f"   Total de imágenes:        {stats['total']}")
    print(f"   Tiempo total:             {total_time:.2f}s")
    print(f"   Tiempo promedio:          {total_time/stats['total']:.2f}s por imagen")
    
    print(f"\n✅ RESULTADOS POR ESTADO:")
    print(f"   Éxito:                    {stats['success']:2d} ({stats['success_rate']:.1f}%)")
    print(f"   Sin placa detectada:      {stats['no_plate_detected']:2d}")
    print(f"   Múltiples placas:         {stats['multiple_plates_detected']:2d}")
    print(f"   Sin texto detectado:      {stats['no_text_detected']:2d}")
    print(f"   Baja confianza:           {stats['low_confidence']:2d}")
    print(f"   Formato inválido:         {stats['invalid_format']:2d}")
    print(f"   Errores:                  {stats['errors']:2d}")
    
    if stats['success'] > 0:
        print(f"\n📈 MÉTRICAS DE ÉXITO:")
        print(f"   Confianza promedio:       {stats['avg_confidence']:.1%}")
    
    # Listar placas exitosas
    successful = [(name, result) for name, result in results.items() 
                  if result.status == PipelineStatus.SUCCESS]
    
    if successful:
        print(f"\n🎉 PLACAS RECONOCIDAS EXITOSAMENTE ({len(successful)}):")
        print_separator("-")
        successful_sorted = sorted(successful, key=lambda x: x[1].confidence, reverse=True)
        
        for image_name, result in successful_sorted:
            print(f"   📄 {image_name:30s} → {result.plate_text:12s} "
                  f"(OCR: {result.confidence:.1%}, Det: {result.detection_confidence:.1%})")
    
    # Listar problemas comunes
    print(f"\n")
    print_separator("-")
    print("⚠️  ANÁLISIS DE PROBLEMAS COMUNES:")
    print_separator("-")
    
    problems = {
        PipelineStatus.NO_PLATE_DETECTED: [],
        PipelineStatus.MULTIPLE_PLATES_DETECTED: [],
        PipelineStatus.NO_TEXT_DETECTED: [],
        PipelineStatus.LOW_CONFIDENCE: [],
        PipelineStatus.INVALID_FORMAT: [],
        PipelineStatus.ERROR: []
    }
    
    for image_name, result in results.items():
        if result.status in problems:
            problems[result.status].append(image_name)
    
    if problems[PipelineStatus.NO_PLATE_DETECTED]:
        print(f"\n🚫 Sin placa detectada ({len(problems[PipelineStatus.NO_PLATE_DETECTED])} imágenes):")
        print("   Causa: El detector no encontró ninguna placa en la imagen")
        print("   Solución: Verificar que la imagen contenga una placa visible")
        for img in problems[PipelineStatus.NO_PLATE_DETECTED]:
            print(f"      • {img}")
    
    if problems[PipelineStatus.MULTIPLE_PLATES_DETECTED]:
        print(f"\n⚠️  Múltiples placas ({len(problems[PipelineStatus.MULTIPLE_PLATES_DETECTED])} imágenes):")
        print("   Causa: Se detectaron varias placas en la misma imagen")
        print("   Solución: El usuario debe tomar otra foto con solo una placa")
        for img in problems[PipelineStatus.MULTIPLE_PLATES_DETECTED]:
            print(f"      • {img}")
    
    if problems[PipelineStatus.NO_TEXT_DETECTED]:
        print(f"\n📝 Sin texto detectado ({len(problems[PipelineStatus.NO_TEXT_DETECTED])} imágenes):")
        print("   Causa: El OCR no pudo extraer texto de la placa detectada")
        print("   Solución: Mejorar calidad de imagen o iluminación")
        for img in problems[PipelineStatus.NO_TEXT_DETECTED]:
            print(f"      • {img}")
    
    if problems[PipelineStatus.LOW_CONFIDENCE]:
        print(f"\n📉 Baja confianza ({len(problems[PipelineStatus.LOW_CONFIDENCE])} imágenes):")
        print("   Causa: El OCR detectó texto pero con baja confianza")
        print("   Solución: Verificar iluminación y nitidez")
        for img in problems[PipelineStatus.LOW_CONFIDENCE]:
            result = results[img]
            conf_str = f"{result.confidence:.1%}" if result.confidence else "N/A"
            print(f"      • {img} (confianza: {conf_str})")
    
    if problems[PipelineStatus.INVALID_FORMAT]:
        print(f"\n❌ Formato inválido ({len(problems[PipelineStatus.INVALID_FORMAT])} imágenes):")
        print("   Causa: El texto detectado no parece ser una placa válida")
        print("   Solución: Verificar que solo la placa esté en la imagen")
        for img in problems[PipelineStatus.INVALID_FORMAT]:
            result = results[img]
            text = result.plate_text if result.plate_text else "N/A"
            print(f"      • {img} (texto: '{text}')")
    
    if problems[PipelineStatus.ERROR]:
        print(f"\n💥 Errores ({len(problems[PipelineStatus.ERROR])} imágenes):")
        print("   Causa: Error técnico durante el procesamiento")
        for img in problems[PipelineStatus.ERROR]:
            print(f"      • {img}")
    
    # CONCLUSIONES
    print(f"\n")
    print_separator("=")
    print("✅ CONCLUSIONES DEL PIPELINE")
    print_separator("=")
    
    print(f"\n📊 Tasa de éxito total: {stats['success_rate']:.1f}%")
    print(f"   → {stats['success']} de {stats['total']} imágenes procesadas exitosamente")
    
    if stats['success_rate'] >= 80:
        print("\n🎉 Excelente! El pipeline está funcionando muy bien")
    elif stats['success_rate'] >= 60:
        print("\n✅ Bien! El pipeline funciona correctamente")
        print("   Considera mejorar la calidad de las imágenes de prueba")
    elif stats['success_rate'] >= 40:
        print("\n⚠️  Moderado. El pipeline funciona pero hay margen de mejora")
        print("   Revisa las imágenes con problemas")
    else:
        print("\n❌ Baja tasa de éxito. Revisa:")
        print("   • Calidad de las imágenes de prueba")
        print("   • Configuración del modelo")
        print("   • Umbrales de confianza")
    
    # Integración con backend
    print(f"\n📱 PARA INTEGRACIÓN CON BACKEND:")
    print("   1. El pipeline devuelve un PipelineResult con:")
    print("      • status: Estado del procesamiento")
    print("      • plate_text: Texto de la placa (si exitoso)")
    print("      • message: Mensaje para mostrar al usuario")
    print("      • confidence: Nivel de confianza")
    
    print("\n   2. Uso recomendado:")
    print("      ```python")
    print("      pipeline = PlateRecognitionPipeline()")
    print("      result = pipeline.process_image(image_path)")
    print("      ")
    print("      if result.is_success():")
    print("          # Llamar a la API con result.plate_text")
    print("          vehicle_data = api.get_vehicle_info(result.plate_text)")
    print("      else:")
    print("          # Mostrar result.message al usuario")
    print("          return {'error': result.message}")
    print("      ```")
    
    print("\n   3. Estados a manejar en el frontend:")
    print("      • SUCCESS → Continuar con consulta API")
    print("      • NO_PLATE_DETECTED → Pedir nueva foto")
    print("      • MULTIPLE_PLATES_DETECTED → Pedir foto más centrada")
    print("      • NO_TEXT_DETECTED → Mejorar iluminación")
    print("      • LOW_CONFIDENCE → Opción de reintento o confirmación manual")
    print("      • INVALID_FORMAT → Verificar que es una placa")
    print("      • ERROR → Mensaje técnico de error")
    
    print(f"\n📂 Archivos generados:")
    print(f"   • Placas detectadas en: {settings.OUTPUT_DIR}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()