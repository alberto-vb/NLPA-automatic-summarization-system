import pdfplumber
import re
import json
from transformers import pipeline
from typing import List
import os
from textwrap import wrap
import string

# ---------- UTILIDADES ----------
def normalize_text(text: str) -> str:
    """
    Convierte el texto a minúsculas, elimina espacios extra y remueve la puntuación,
    para facilitar la comparación.
    """
    text = text.lower().strip()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def remove_duplicates_strict(seq: list) -> list:
    """
    Elimina duplicados utilizando la versión normalizada para la comparación.
    Mantiene el orden original.
    """
    seen = set()
    result = []
    for x in seq:
        norm = normalize_text(x)
        if norm not in seen:
            seen.add(norm)
            result.append(x)
    return result

def maybe_reverse_text(text: str) -> str:
    """
    Detecta si el texto parece estar invertido (por ejemplo, si al invertir los primeros 50 caracteres
    se encuentran cadenas como 'http' o 'boe') y, de ser así, revierte el texto.
    """
    test_str = text[:50]
    reversed_test = test_str[::-1]
    if "http" in reversed_test.lower() or "boe" in reversed_test.lower():
        print("Detectado texto invertido. Revirtiendo el texto...")
        return text[::-1]
    return text

# ---------- LECTURA DE PDF ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    text = maybe_reverse_text(text)
    return text

# ---------- DIVISIÓN EN PÁRRAFOS ----------
def split_into_paragraphs(text: str) -> List[str]:
    # Primero intenta dividir por doble salto de línea; si no hay suficientes, usa salto simple.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

# ---------- FILTRO DE PÁRRAFOS ----------
def find_paragraphs_with_dates(paragraphs: List[str]) -> List[str]:
    # Se buscan dos formatos de fecha: "21 de diciembre de 2007" y "21/12/2007"
    regex_date1 = r"\d{1,2}\s+de\s+\w+\s+de\s+\d{4}"
    regex_date2 = r"\d{1,2}/\d{1,2}/\d{4}"
    return [p for p in paragraphs if re.search(regex_date1, p, re.IGNORECASE) or re.search(regex_date2, p)]

def find_paragraphs_with_amounts(paragraphs: List[str]) -> List[str]:
    # Busca cantidades en formatos como "1.234,56 €" o "1234,56 €"
    regex_amount = r"\d{1,3}(?:\.\d{3})*,\d{2}\s?€"
    return [p for p in paragraphs if re.search(regex_amount, p)]

def find_paragraphs_with_requirements(paragraphs: List[str]) -> List[str]:
    keywords = ["requisito", "nota media", "haber superado", "matriculado", "mínimo de créditos"]
    return [p for p in paragraphs if any(kw in p.lower() for kw in keywords)]

# ---------- CONSTRUCCIÓN DEL PROMPT ----------
def build_summary_input_v2(date_paragraphs: List[str],
                             amount_paragraphs: List[str],
                             requirement_paragraphs: List[str]) -> str:
    prompt = (
        "Elabora un resumen conciso y claro en lenguaje natural, orientado a estudiantes, "
        "destacando los plazos de solicitud, las cuantías económicas y los requisitos académicos de la convocatoria. "
        "Omite repeticiones y redundancias. A continuación se muestra la información extraída:\n\n"
    )
    if date_paragraphs:
        dates = remove_duplicates_strict(date_paragraphs)
        prompt += "Plazos:\n" + "\n".join(dates) + "\n\n"
    if amount_paragraphs:
        amounts = remove_duplicates_strict(amount_paragraphs)
        prompt += "Cuantías:\n" + "\n".join(amounts) + "\n\n"
    if requirement_paragraphs:
        reqs = remove_duplicates_strict(requirement_paragraphs)
        prompt += "Requisitos:\n" + "\n".join(reqs) + "\n\n"
    return prompt.strip()

# ---------- RESUMIDOR ----------
# Inicializamos el pipeline con un límite mayor y early_stopping activado.
summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=1024,
    early_stopping=True
)

def generate_summary(text: str) -> str:
    return summarizer(text)[0]["generated_text"]

def hierarchical_summarize(texts: List[str], group_size: int = 10) -> str:
    """
    Resume una lista de textos dividiéndolos en grupos de tamaño 'group_size',
    resumiendo cada grupo y luego combinando los resúmenes intermedios en un resumen final.
    """
    groups = [texts[i:i+group_size] for i in range(0, len(texts), group_size)]
    group_summaries = []
    for group in groups:
        group_prompt = "Resume los siguientes fragmentos relacionados con una convocatoria de beca:\n" + "\n".join(group)
        summary = summarizer(group_prompt)[0]["generated_text"]
        group_summaries.append(summary)
    if len(group_summaries) == 1:
        return group_summaries[0]
    else:
        final_prompt = "Resume los siguientes resúmenes relacionados con una convocatoria de beca:\n" + "\n".join(group_summaries)
        final_summary = summarizer(final_prompt)[0]["generated_text"]
        return final_summary

def generate_chunked_summary(long_text: str, max_chars: int = 500) -> str:
    """
    Divide el texto en fragmentos de hasta 'max_chars' caracteres y resume cada uno (procesándolos en batch).
    Si se generan muchos mini resúmenes, se aplica una estrategia jerárquica para combinarlos.
    """
    chunks = wrap(long_text, max_chars)
    if not chunks:
        return ""
    print(f"Número de chunks generados: {len(chunks)}")
    
    # Procesar los fragmentos en batch
    mini_outputs = summarizer(chunks, batch_size=4)
    mini_summaries = [output["generated_text"] for output in mini_outputs]
    
    # Si hay más de 10 mini resúmenes, aplicar resumen jerárquico
    if len(mini_summaries) > 10:
        return hierarchical_summarize(mini_summaries, group_size=10)
    else:
        resumen_global_prompt = (
            "A continuación se presentan varios fragmentos resumidos de una convocatoria de beca.\n"
            "Resume los aspectos clave para un estudiante (plazos, requisitos, cuantía):\n\n" +
            "\n".join(mini_summaries)
        )
        return summarizer(resumen_global_prompt)[0]["generated_text"]

# ---------- MAIN ----------
if __name__ == "__main__":
    pdfs = [
        'corpus/ayudas_20-21.pdf',
        'corpus/ayudas_21-22.pdf',
        'corpus/ayudas_22-23.pdf',
        'corpus/ayudas_23-24.pdf',
        'corpus/ayudas_24-25.pdf',
    ]
    resultados = []
    for pdf in pdfs:
        print(f"\nProcesando: {pdf}")
        try:
            raw_text = extract_text_from_pdf(pdf)
            print("Vista previa del texto extraído:", raw_text[:200])
            
            paragraphs = split_into_paragraphs(raw_text)
            print("Número de párrafos extraídos:", len(paragraphs))
            if paragraphs:
                print("Primeros párrafos:", paragraphs[:3])
            
            date_paragraphs = find_paragraphs_with_dates(paragraphs)
            amount_paragraphs = find_paragraphs_with_amounts(paragraphs)
            requirement_paragraphs = find_paragraphs_with_requirements(paragraphs)
            
            print("Párrafos con fechas:", date_paragraphs[:5])
            print("Párrafos con cuantías:", amount_paragraphs[:5])
            print("Párrafos con requisitos:", requirement_paragraphs[:5])
            
            summary_input = build_summary_input_v2(date_paragraphs, amount_paragraphs, requirement_paragraphs)
            print("Prompt para el resumen:")
            print(summary_input)
            
            resumen = generate_chunked_summary(summary_input)
            print("Resumen generado:")
            print(resumen)
            
            resultados.append({
                "documento": pdf,
                "resumen": resumen
            })
            
            # Guardar el resumen en un archivo .txt
            base_name = os.path.basename(pdf).replace('.pdf', '_resumen.txt')
            txt_path = os.path.join("corpus", base_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(resumen)
        except Exception as e:
            print(f"Error procesando {pdf}: {e}")
    
    # Guardar todos los resúmenes en un JSON
    with open("corpus/resumenes_focalizados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    # Crear un resumen general a partir de los resúmenes individuales
    resumenes_texto = "\n\n".join(
        f"{os.path.basename(r['documento'])}:\n{r['resumen']}" for r in resultados
    )
    resumen_general_prompt = (
        "A continuación se presentan resúmenes de distintas convocatorias de becas.\n"
        "Genera un resumen general para los estudiantes, indicando patrones comunes, "
        "requisitos frecuentes, cuantías típicas y fechas aproximadas de solicitud.\n\n" +
        resumenes_texto
    )
    resumen_general = generate_summary(resumen_general_prompt)
    print("\nResumen general generado:")
    print(resumen_general)
    
    with open("corpus/resumen_general.txt", "w", encoding="utf-8") as f:
        f.write(resumen_general)
