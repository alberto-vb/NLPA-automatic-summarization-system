import json
import re
import os

def parse_requisitos(text: str) -> dict:
    """
    Extrae únicamente la información de matriculacion_minima (bullets 1.º), 2.º), etc.).
    Se ELIMINA 'nota_minima_master' y no se genera.
    
    Devuelve algo como:
    {
      "matriculacion_minima": { ... }
    }
    """
    results = {
        "matriculacion_minima": {}
    }

    pattern_bullets = re.compile(
        r'(?s)(\d{1,2}\.\s*º\))(.*?)(?=(\d{1,2}\.\s*º\))|$)',
        flags=re.MULTILINE
    )

    for match in pattern_bullets.finditer(text):
        bullet = match.group(1).strip()   # "1.º)"
        content = match.group(2).strip()

        lines = content.splitlines()
        if not lines:
            results["matriculacion_minima"][bullet] = ""
            continue

        first_line = lines[0].strip()
        colonmatch = re.match(r'^(.*?)\:\s*(.*)$', first_line)
        if colonmatch:
            title = colonmatch.group(1).strip()
            remainder = colonmatch.group(2).strip()
            if len(lines) > 1:
                remainder += "\n" + "\n".join(lines[1:])
            results["matriculacion_minima"][title] = remainder
        else:
            results["matriculacion_minima"][bullet] = content

    return results


def parse_cuantias(text: str) -> dict:
    results = {
        "beca_matricula": None,
        "fija_renta": None,
        "fija_residencia": None,
        "beca_basica_normal": None,
        "beca_basica_fp_basico": None,
        "variable_minima": None,
        "porcentajes_por_rama": {}
    }

    match_matricula = re.search(
        r"beca de matrícula.*?(cubrirá|cubrir)(.*?)(\d[\d\.,]*\s*(€|euros))?",
        text, flags=re.IGNORECASE|re.DOTALL
    )
    if match_matricula:
        results["beca_matricula"] = "Gratuidad de los créditos de primera matrícula"

    match_renta = re.search(
        r"cuantía fija ligada a la renta.*?(\d[\d\.,]+)\s*(euros|€)",
        text, flags=re.IGNORECASE
    )
    if match_renta:
        results["fija_renta"] = match_renta.group(1).replace(".", "")

    match_residencia = re.search(
        r"cuantía fija ligada a la residencia.*?(\d[\d\.,]+)\s*(euros|€)",
        text, flags=re.IGNORECASE
    )
    if match_residencia:
        results["fija_residencia"] = match_residencia.group(1).replace(".", "")

    block_basica = re.search(
        r"Beca básica:\s*(\d[\d\.,]+)\s*(?:euros|€).*?esta\s*cuantía\s*será\s*de\s*(\d[\d\.,]+)\s*(?:euros|€)",
        text, flags=re.IGNORECASE|re.DOTALL
    )
    if block_basica:
        results["beca_basica_normal"] = block_basica.group(1)
        results["beca_basica_fp_basico"] = block_basica.group(2)
    else:
        solo_basica = re.search(
            r"Beca básica:\s*(\d[\d\.,]+)\s*(?:euros|€)",
            text, flags=re.IGNORECASE
        )
        if solo_basica:
            results["beca_basica_normal"] = solo_basica.group(1)

    match_variable = re.search(
        r"cuantía variable.*?(?:importe mínimo.*?)(\d[\d\.,]+)\s*(euros|€)",
        text, flags=re.IGNORECASE|re.DOTALL
    )
    if match_variable:
        results["variable_minima"] = match_variable.group(1)

    patron_ramas = re.compile(
        r"(Artes y Humanidades|Ciencias Sociales y Jurídicas|Ciencias de la Salud|Ingeniería o Arquitectura[/\w\s]*técnicas|Ciencias).*?(\d{1,3})\%",
        flags=re.IGNORECASE
    )
    for (rama_raw, porc_str) in patron_ramas.findall(text):
        porc_int = int(porc_str)
        rama_lower = rama_raw.lower()

        if "artes y humanidades" in rama_lower:
            results["porcentajes_por_rama"]["artes_humanidades"] = porc_int
        elif "sociales y jurídicas" in rama_lower:
            results["porcentajes_por_rama"]["ciencias_sociales_juridicas"] = porc_int
        elif "salud" in rama_lower:
            results["porcentajes_por_rama"]["ciencias_de_la_salud"] = porc_int
        elif "ingeniería" in rama_lower or "arquitectura" in rama_lower:
            results["porcentajes_por_rama"]["ingenieria_arquitectura_tecnicas"] = porc_int
        elif "ciencias" in rama_lower:
            results["porcentajes_por_rama"]["ciencias"] = porc_int

    return results


def parse_excelencia(text: str) -> dict:
    """
    Cambiamos 'excelencia_min' -> 'excelencia_cuantia_min'
    y 'excelencia_max' -> 'excelencia_cuantia_max'
    """
    results = {
        "nota_minima_excelencia": None,
        "excelencia_cuantia_min": None,
        "excelencia_cuantia_max": None
    }

    m_nota_min = re.search(
        r"se\s+requerirá.*?(\d[\d\.,]+)\s*(?:puntos|o superior)",
        text, flags=re.IGNORECASE
    )
    if m_nota_min:
        results["nota_minima_excelencia"] = m_nota_min.group(1)

    m_ex = re.search(
        r"excelencia\s+académica:\s*entre\s+(\d[\d\.,]+)\s+y\s+(\d[\d\.,]+)\s+euros",
        text, flags=re.IGNORECASE
    )
    if m_ex:
        results["excelencia_cuantia_min"] = m_ex.group(1).replace('.', '').replace(',', '')
        results["excelencia_cuantia_max"] = m_ex.group(2).replace('.', '').replace(',', '')

    return results


def parse_plazo(text: str) -> dict:
    """
    Solo 'plazo_presentacion_fin' y no 'plazo_presentacion_inicio'
    """
    results = {
        "plazo_presentacion_fin": None
    }
    match = re.search(
        r"(?:plazo.*?hasta\s+el\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4}))",
        text, flags=re.IGNORECASE
    )
    if match:
        results["plazo_presentacion_fin"] = match.group(1)
    return results


def parse_solicitud(text: str) -> dict:
    """
    Solo 'donde_presentar' y no 'fecha_limite'
    """
    result = {
        "donde_presentar": None
    }
    match_donde = re.search(
        r"(solicitudes?\sse\spresentarán.*?\.)",
        text, flags=re.IGNORECASE
    )
    if match_donde:
        result["donde_presentar"] = match_donde.group(1).strip()
    return result


def main():
    input_folder = "data/sections"
    output_folder = "data/parsed"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".json"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"[Parseando información...] Procesando {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        req_text  = data.get("requisitos",[{"content":""}])[0]["content"]
        cuan_text = data.get("cuantias",[{"content":""}])[0]["content"]
        plazo_text= data.get("plazo",[{"content":""}])[0]["content"]
        sol_text  = data.get("solicitud",[{"content":""}])[0]["content"]
        ex_text   = data.get("excelencia",[{"content":""}])[0]["content"]

        # Parseamos
        req_parsed   = parse_requisitos(req_text)
        cuan_parsed  = parse_cuantias(cuan_text)
        plazo_parsed = parse_plazo(plazo_text)
        sol_parsed   = parse_solicitud(sol_text)
        ex_parsed    = parse_excelencia(ex_text)

        # Construimos final_info
        final_info = {
            "requisitos": req_parsed,     # sin nota_minima_master
            "cuantias": cuan_parsed,
            "excelencia": ex_parsed,      # excel_cuantia_min & excel_cuantia_max
            "plazo": plazo_parsed,
            "solicitud": sol_parsed
        }

        # Mover porcentajes_por_rama de cuantias a requisitos
        por_rama = final_info["cuantias"].pop("porcentajes_por_rama", None)
        if por_rama:
            final_info["requisitos"]["porcentajes_por_rama"] = por_rama

        # Guardamos
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(final_info, out, indent=2, ensure_ascii=False)

        print(f"Archivo parseado guardado en: {output_path}")


if __name__ == "__main__":
    main()
