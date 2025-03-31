import pdfplumber
import re
import json
from transformers import pipeline
from textwrap import wrap

# ---------- EXTRACCIÓN DE TEXTO ----------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and concatenates text from all pages of the PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---------- ENTIDADES NOMBRADAS ----------

def extract_entities(text: str, chunk_size: int = 400) -> list:
    """Uses NER pipeline with chunking to avoid token overflow."""
    ner_pipeline = pipeline(
        "ner",
        model="mrm8488/bert-spanish-cased-finetuned-ner",
        aggregation_strategy="simple"
    )
    chunks = wrap(text, chunk_size)
    all_entities = []
    for chunk in chunks:
        all_entities.extend(ner_pipeline(chunk))
    return all_entities


# ---------- EXTRACCIÓN REGLADA DE CAMPOS ----------

def extract_amounts(text: str):
    return re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?€", text)

def extract_deadlines(text: str):
    return re.findall(r"hasta el \d{1,2} de \w+ de \d{4}", text, flags=re.IGNORECASE)

def extract_income_thresholds(text: str):
    return re.findall(r"(renta.*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€)", text, flags=re.IGNORECASE)

def extract_education_levels(text: str):
    niveles = ["educación secundaria", "bachillerato", "formación profesional", "grado", "máster", "universidad"]
    return [nivel for nivel in niveles if nivel in text.lower()]

def extract_academic_requirements(text: str):
    pattern = r"(haber\s.+?créditos|nota\smedia\s.+?\d,\d+|superar\s.+?%)"
    return re.findall(pattern, text, flags=re.IGNORECASE)

def extract_legal_references(text: str):
    return re.findall(r"(Real Decreto\s\d+/\d+|BOE\s(núm\.|\d+).+?\d{4})", text)

def extract_submission_urls(text: str):
    return re.findall(r"https?://\S+", text)


# ---------- ESTRUCTURAR INFORMACIÓN ----------

def structure_information(text: str, entities: list, doc_name: str) -> dict:
    education_levels = extract_education_levels(text)
    scholarship_amounts = extract_amounts(text)
    income_thresholds = [i[1] for i in extract_income_thresholds(text)]
    academic_requirements = extract_academic_requirements(text)
    application_deadline = extract_deadlines(text)
    legal_references = [ref[0] for ref in extract_legal_references(text)]
    submission_platform = extract_submission_urls(text)

    info = {
        "documento": doc_name,
        "autoridad_emisora": "",
        "niveles_educativos": sorted(set(education_levels)),
        "importes_beca": sorted(set(scholarship_amounts)),
        "umbrales_de_ingresos": sorted(set(income_thresholds)),
        "requisitos_académicos": sorted(set(academic_requirements)),
        "fecha_limite": sorted(set(application_deadline)),
        "legal_references": sorted(set(legal_references)),
        "plataforma": sorted(set(submission_platform))
    }

    # Tomamos el primer ORG como organismo convocante
    organizations = [e["word"] for e in entities if e["entity_group"] == "ORG"]
    if organizations:
        info["autoridad_emisora"] = organizations[0]

    return info


# ---------- GENERACIÓN DE RESUMEN ----------

def generate_summary(data: dict):
    summarizer = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)

    prompt = f"""
    Genera un resumen claro y conciso para estudiantes sobre la beca publicada en {data['documento']}.

    Organismo: {data['autoridad_emisora']}
    Niveles educativos: {", ".join(data['niveles_educativos'])}
    Cuantía: {", ".join(data['importes_beca'])}
    Umbral de renta: {", ".join(data['umbrales_de_ingresos'])}
    Requisitos académicos: {", ".join(data['requisitos_académicos'])}
    Plazo de solicitud: {", ".join(data['fecha_limite'])}
    Más info: {", ".join(data['plataforma'])}
    """

    summary = summarizer(prompt)[0]["generated_text"]
    return summary


# ---------- MAIN ----------

if __name__ == "__main__":
    pdfs = [
        'corpus/ayudas_20-21.pdf',
        'corpus/ayudas_21-22.pdf',
        'corpus/ayudas_22-23.pdf',
        'corpus/ayudas_23-24.pdf',
        'corpus/ayudas_24-25.pdf',
    ]

    all_info = []

    for pdf in pdfs:
        print(f"\nProcesando: {pdf}")
        try:
            raw_text = extract_text_from_pdf(pdf)
        except Exception as e:
            print(f"Error al leer {pdf}: {e}")
            continue

        entities = extract_entities(raw_text)
        structured_data = structure_information(raw_text, entities, pdf)

        # Guardar JSON individual
        json_path = f"corpus/{pdf.split('/')[-1].replace('.pdf', '_info.json')}"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        # Generar resumen
        resumen = generate_summary(structured_data)
        print("Resumen generado:")
        print(resumen)

        # También puedes guardar todos los datos para análisis conjunto
        structured_data["summary"] = resumen
        all_info.append(structured_data)

    # Guardar resumen conjunto
    with open("corpus/resumenes_becas.json", "w", encoding="utf-8") as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)