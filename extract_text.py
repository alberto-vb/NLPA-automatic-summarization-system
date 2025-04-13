import fitz  # PyMuPDF
import os
import re

INPUT_FOLDER = "data/corpus"
OUTPUT_FOLDER = "data/extracted"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_paragraph(paragraph: str) -> str:
    """
    Limpia un párrafo uniendo líneas y eliminando saltos innecesarios.
    """
    lines = paragraph.split("\n")
    result = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Si hay un guion al final (corte de palabra), lo quitamos.
        if line.endswith("-"):
            line = line[:-1]

        if result:
            if not result.endswith((".", ":", ";", "?", "¡", "!", "¿")):
                result += " " + line
            else:
                result += " " + line
        else:
            result = line

    return result


def remove_signature_lines(text: str) -> str:
    """
    Elimina todas las líneas que contengan referencias de CSV, validación,
    firmantes, fechas o notas.
    """
    # Separamos el texto en líneas
    lines = text.split("\n")
    filtered_lines = []
    
    for line in lines:
        # Si la línea incluye cualquiera de estas cadenas, la descartamos
        if ( 
            "CSV : GEN-" in line 
            or "DIRECCIÓN DE VALIDACIÓN" in line 
            or "FIRMANTE(" in line 
            or "FECHA :" in line 
            or "NOTAS :" in line
        ):
            continue
        filtered_lines.append(line)
    
    # Unimos las líneas filtradas de nuevo en un solo texto
    return "\n".join(filtered_lines)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae el texto de un archivo PDF, aplicando la limpieza de párrafos
    y filtrando líneas innecesarias.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error abriendo {pdf_path}: {e}")
        return ""

    full_text = []

    for page in doc:
        # Primero obtenemos el texto de la página
        page_text = page.get_text("text")

        # Eliminamos las líneas que contengan firmas, CSV, etc.
        page_text = remove_signature_lines(page_text)

        # Luego separamos por párrafos (doble salto de línea como heurística)
        paragraphs = page_text.split("\n\n")

        # Limpiamos cada párrafo uniendo líneas
        clean_paragraphs = [clean_paragraph(p) for p in paragraphs if p.strip()]
        full_text.extend(clean_paragraphs)

    return "\n\n".join(full_text).strip()


def process_all_pdfs():
    """
    Procesa todos los PDFs en INPUT_FOLDER y guarda la versión de texto limpio
    en OUTPUT_FOLDER.
    """
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".pdf", ".txt"))

            print(f"Extrayendo texto de {filename}...")
            text = extract_text_from_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    process_all_pdfs()
