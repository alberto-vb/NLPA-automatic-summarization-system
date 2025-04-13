import os
import json
import re

INPUT_FOLDER = "data/extracted"
OUTPUT_FOLDER = "data/sections"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1) Definimos las secciones y sus "sinónimos" o encabezados que disparan cada sección
SECTIONS = {
    "requisitos": ["REQUISITOS", "REQUISITOS DE LOS SOLICITANTES"],
    "cuantias":   ["CUANTÍAS", "CUANTÍA", "IMPORTE", "artículo"],
    "plazo":      ["PLAZO", "PLAZO DE SOLICITUD", "PLAZO DE PRESENTACIÓN"],
    "solicitud":  ["SOLICITUD", "FORMA DE PRESENTACIÓN", "CÓMO SOLICITAR"],
    # Aquí creamos la nueva sección "excelencia":
    "excelencia": ["EXCELENCIA", "EXCELENCIA ACADÉMICA", "CUANTÍA FIJA LIGADA A LA EXCELENCIA"]
}

def locate_sections(text: str) -> dict:
    """
    Segmenta un texto según SECTIONS, unificando todo lo que pertenezca a una misma 
    sección en un único bloque. Por ejemplo, si 'excelencia' tiene sinónimos 
    ["EXCELENCIA", "EXCELENCIA ACADÉMICA"], 
    se agrupará todo el contenido que esté bajo esos encabezados en found_sections["excelencia"].
    """
    # Inicializamos el resultado: para cada sección, creamos un array con un único dict
    # con "heading" y "content". 
    found_sections = {}
    for section_key in SECTIONS.keys():
        found_sections[section_key] = [{
            "heading": section_key,  # El "título" interno de la sección
            "content": ""
        }]

    # 2) Construimos un patrón con OR de todos los sinónimos de todas las secciones
    #    Si alguno de tus títulos contiene caracteres especiales, usa re.escape(t).
    all_titles = []
    for synonyms in SECTIONS.values():
        all_titles.extend(synonyms)

    # Si necesitas una regex avanzada para "artículo" (ej. r"artículo\s+\d+(\.)?") hazlo aparte
    pattern = "|".join(all_titles)

    # 3) Buscamos cada coincidencia (heading) en el texto con ignorecase
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        # Bloque de texto desde este heading hasta el siguiente
        section_text = text[start:end].strip()

        # Determinamos cuál de las SECTIONS encaja
        heading_text = match.group(0)     # la subcadena exacta capturada
        heading_lower = heading_text.lower()

        # Recorremos SECTIONS para ver a cuál pertenece
        for key, synonyms in SECTIONS.items():
            for syn in synonyms:
                if syn.lower() in heading_lower:
                    # concatenamos
                    found_sections[key][0]["content"] += "\n\n" + section_text
                    break
            else:
                # si no rompemos, no coincide
                continue
            # si hemos roto, ya hallamos la sección
            break

    # Ejemplo adicional:
    # Si quieres filtrar "cuantias" si no contienen euros
    content_lower = found_sections["cuantias"][0]["content"].lower()
    if "€" not in content_lower and "euros" not in content_lower:
        found_sections["cuantias"][0]["content"] = ""

    return found_sections

def process_all_txts():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".txt", ".json"))

            print(f"Extrayendo secciones de {filename}...")
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            sections = locate_sections(text)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, indent=2, ensure_ascii=False)

            print(f"Secciones extraídas → {output_path}")


if __name__ == "__main__":
    process_all_txts()
