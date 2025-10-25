"""
get_plate_data.py
Módulo para consultar RegCheck (regcheck.org.uk) y devolver datos del vehículo a partir de un número de placa.
Uso principal: get_plate_data(plate_number, username=..., country='UK')

Retorna:
    dict con keys:
      - status: "success" | "not_found" | "error"
      - message: str
      - plate: str
      - source: "regcheck"
      - vehicle: dict | None  (parsed vehicleJson if available)
      - raw_xml: str | None    (opcional, raw XML devuelto por la API)
      - owner: None (RegCheck no proporciona dueño por defecto; campo preparado por si se agrega otro proveedor)
      - http_status: int | None
      - query: dict (metadatos de la consulta)
"""

import requests
import xml.etree.ElementTree as ET
import json
from typing import Optional, Dict

REGCHECK_BASE = "https://www.regcheck.org.uk/api/reg.asmx/CheckEcuador"


def _parse_vehicle_json_from_xml(xml_text: str) -> Optional[Dict]:
    """
    Extrae el elemento <vehicleJson> del XML y devuelve el JSON parseado como dict.
    Si no encuentra vehicleJson o el JSON no es decodificable, devuelve None.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # Buscar cualquier elemento llamado 'vehicleJson' en el XML (namespaces ignorados si aparecen)
    vehicle_json_str = None
    # Recorremos en profundidad para evitar problemas con namespaces
    for elem in root.iter():
        tag = elem.tag
        # Normalizar tag a su nombre simple si viene con namespace {..}tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        if tag == 'vehicleJson':
            vehicle_json_str = (elem.text or "").strip()
            break

    if not vehicle_json_str:
        return None

    # El servicio a veces devuelve JSON con claves sin comillas (ejemplo en docs),
    # intentamos primero json.loads directo; si falla intentamos limpieza mínima.
    try:
        return json.loads(vehicle_json_str)
    except Exception:
        # Intento de reparación: reemplazar claves sin comillas por claves entre comillas
        # (Solución heurística: preferir retornar None si no se puede decodificar limpiamente)
        try:
            # Si el string contiene '=>' o similar, no intentar; devolver None
            cleaned = vehicle_json_str.replace("\r", "").replace("\n", "").strip()
            return json.loads(cleaned)
        except Exception:
            return None


def get_plate_data(
    plate_number: str,
    username: str,
    timeout: float = 10.0,
    raw: bool = False
) -> Dict:
    """
    Consulta RegCheck para un número de placa y devuelve un dict normalizado.

    Args:
        plate_number: número de placa a consultar (string)
        username: usuario de RegCheck (necesario)
        timeout: tiempo máximo en segundos para la petición HTTP
        raw: si True, incluye 'raw_xml' en la respuesta

    Returns:
        dict con la estructura descrita arriba.
    """
    # Normalizar entrada
    plate = str(plate_number).strip().upper()
    if not plate:
        return {
            "status": "error",
            "message": "Número de placa vacío",
            "plate": plate,
            "source": "regcheck",
            "vehicle": None,
            "raw_xml": None,
            "owner": None,
            "http_status": None,
            "query": {"endpoint": REGCHECK_BASE, "params": {"RegistrationNumber": plate, "username": username}}
        }

    params = {"RegistrationNumber": plate, "username": username}

    try:
        # RegCheck soporta GET simple: /api/reg.asmx/Check?RegistrationNumber=...&username=...
        resp = requests.get(REGCHECK_BASE, params=params, timeout=timeout)
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Error en la solicitud HTTP: {str(e)}",
            "plate": plate,
            "source": "regcheck",
            "vehicle": None,
            "raw_xml": None,
            "owner": None,
            "http_status": None,
            "query": {"endpoint": REGCHECK_BASE, "params": params}
        }

    http_status = resp.status_code
    raw_xml = resp.text

    if http_status != 200:
        return {
            "status": "error",
            "message": f"Respuesta no OK del servidor (HTTP {http_status})",
            "plate": plate,
            "source": "regcheck",
            "vehicle": None,
            "raw_xml": raw_xml if raw else None,
            "owner": None,
            "http_status": http_status,
            "query": {"endpoint": REGCHECK_BASE, "params": params}
        }

    # Intentar parsear vehicleJson
    vehicle = _parse_vehicle_json_from_xml(raw_xml)

    if not vehicle:
        # No data found or parsing failed. Puede ser placa no encontrada o formato inesperado.
        # Buscamos si hay nodos que indiquen mensaje de error en el XML (ej: <string>..</string> u otros)
        # Intento muy sencillo de extraer texto legible
        try:
            root = ET.fromstring(raw_xml)
            all_text = " ".join([ (el.text or "").strip() for el in root.iter() ])
            # Si aparece 'No data' o 'not found' -> not_found
            low = all_text.lower()
            if "no data" in low or "not found" in low or "not available" in low:
                status = "not_found"
                message = "No se encontraron datos para la placa solicitada."
            else:
                status = "error"
                message = "Respuesta recibida pero no se pudo extraer vehicleJson (posible formato inesperado)."
        except ET.ParseError:
            status = "error"
            message = "Respuesta recibida pero XML malformado y no se pudo parsear."

        return {
            "status": status,
            "message": message,
            "plate": plate,
            "source": "regcheck",
            "vehicle": None,
            "raw_xml": raw_xml if raw else None,
            "owner": None,
            "http_status": http_status,
            "query": {"endpoint": REGCHECK_BASE, "params": params}
        }

    # Éxito: devolvemos objeto vehicle normalizado
    return {
        "status": "success",
        "message": "Datos obtenidos correctamente desde RegCheck",
        "plate": plate,
        "source": "regcheck",
        "vehicle": vehicle,
        "raw_xml": raw_xml if raw else None,
        # Por defecto RegCheck no ofrece datos del propietario por privacidad.
        "owner": None,
        "http_status": http_status,
        "query": {"endpoint": REGCHECK_BASE, "params": params}
    }


# Ejemplo de uso
if __name__ == "__main__":
    # Rellenar con tu username de prueba/producción de RegCheck
    USERNAME = "BryanMoreno11"
    plate = "OBC1226"
    result = get_plate_data(plate, username=USERNAME, raw=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
