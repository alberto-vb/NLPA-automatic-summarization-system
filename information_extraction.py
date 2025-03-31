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
    # Original pattern
    original_amounts = re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?€", text)
    # New pattern with range format (1.000-2.000 €)
    range_amounts = re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?(?:\s?-\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?)?\s?€", text)
    # New pattern for annual amounts
    annual_amounts = re.findall(r"(?:anual(?:es)?|por año|por curso).*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€", text, flags=re.IGNORECASE)
    
    # Combine all amounts, removing duplicates
    all_amounts = set(original_amounts + range_amounts)
    if annual_amounts:
        all_amounts.update([amt + " € (anual)" for amt in annual_amounts])
        
    return list(all_amounts)

def extract_deadlines(text: str):
    # Original pattern
    original_deadlines = re.findall(r"hasta el \d{1,2} de \w+ de \d{4}", text, flags=re.IGNORECASE)
    
    # New patterns for additional date formats
    extended_deadlines = re.findall(r"(?:hasta|antes del|fin(?:aliza)?(?:\sel)?|término)(?:\sel)?\s\d{1,2}\s?(?:de)?\s?\w+(?:\sde)?\s?\d{4}", text, flags=re.IGNORECASE)
    
    # Pattern for deadline with specific time
    time_deadlines = re.findall(r"(?:hasta|antes del|fin(?:aliza)?(?:\sel)?|término)(?:\sel)?\s\d{1,2}\s?(?:de)?\s?\w+(?:\sde)?\s?\d{4}(?:\sa\slas)?\s\d{1,2}(?::\d{2})?\s?(?:horas|h\.?)", text, flags=re.IGNORECASE)
    
    # Pattern for application period
    period_deadlines = re.findall(r"(?:periodo|plazo)(?:\sde\ssolicitud|\sde\spresentación)(?:\scomprendido|\sestablecido)?(?:\sentre|\sdesde)\sel\s\d{1,2}\s?(?:de)?\s?\w+(?:\sde)?\s?\d{4}(?:\s(?:y|hasta|al)\sel\s\d{1,2}\s?(?:de)?\s?\w+(?:\sde)?\s?\d{4})?", text, flags=re.IGNORECASE)
    
    return list(set(original_deadlines + extended_deadlines + time_deadlines + period_deadlines))

def extract_income_thresholds(text: str):
    # Original pattern
    original_thresholds = re.findall(r"(renta.*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€)", text, flags=re.IGNORECASE)
    
    # New pattern for family income thresholds
    family_thresholds = re.findall(r"(?:umbral|límite)(?:\sde)?(?:\s\w+){0,3}(?:\sfamiliar|\sper\scápita|\spersonal).*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€", text, flags=re.IGNORECASE)
    
    # Pattern for income thresholds by family members
    members_thresholds = re.findall(r"(?:familia|miembros|personas)(?:\sde)?(?:\s\d+\smiembros|\s\w+).*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€", text, flags=re.IGNORECASE)
    
    # Combine all thresholds
    all_thresholds = original_thresholds
    if family_thresholds:
        all_thresholds.extend([("umbral familiar " + ft, ft) for ft in family_thresholds])
    if members_thresholds:
        all_thresholds.extend([("umbral por miembros " + mt, mt) for mt in members_thresholds])
        
    return all_thresholds

def extract_education_levels(text: str):
    # Original list of education levels
    niveles = ["educación secundaria", "bachillerato", "formación profesional", "grado", "máster", "universidad"]
    
    # Add more specific education levels
    niveles_extendidos = [
        "educación infantil", "educación primaria", "eso", "educación secundaria obligatoria",
        "fp básica", "fp grado medio", "fp grado superior", "ciclo formativo", 
        "estudios superiores", "doctorado", "postgrado", "enseñanzas artísticas",
        "enseñanzas deportivas", "idiomas", "escuela oficial de idiomas"
    ]
    
    # Combine all education levels
    all_niveles = niveles + niveles_extendidos
    
    return [nivel for nivel in all_niveles if re.search(r'\b' + re.escape(nivel) + r'\b', text.lower())]

def extract_academic_requirements(text: str):
    # Original pattern
    original_requirements = re.findall(r"(haber\s.+?créditos|nota\smedia\s.+?\d,\d+|superar\s.+?%)", text, flags=re.IGNORECASE)
    
    # New patterns for academic requirements
    passing_reqs = re.findall(r"(?:haber aprobado|tener aprobad[oa]s|superar)(?:\sun\smínimo\sde)?\s(?:\w+\s){0,3}(?:\d+\s)?(?:asignaturas|créditos|ECTS|materias)", text, flags=re.IGNORECASE)
    
    grade_reqs = re.findall(r"(?:nota\s(?:media|mínima))(?:\srequerida|\snecesaria|\sexigida)?(?:\sde|\s:)?\s\d+(?:[,.]\d+)?", text, flags=re.IGNORECASE)
    
    continuation_reqs = re.findall(r"(?:estar\smatriculado|matricularse)(?:\sen|\sde)(?:\sun\smínimo\sde)?\s(?:\w+\s){0,3}(?:\d+\s)?(?:asignaturas|créditos|ECTS|materias)", text, flags=re.IGNORECASE)
    
    all_requirements = original_requirements + passing_reqs + grade_reqs + continuation_reqs
    
    return list(set(all_requirements))

def extract_legal_references(text: str):
    # Original pattern
    original_references = re.findall(r"(Real Decreto\s\d+/\d+|BOE\s(núm\.|\d+).+?\d{4})", text)
    
    # New patterns for legal references
    law_references = re.findall(r"(?:Ley(?:\sOrgánica)?|Orden(?:\sMinisterial)?|Resolución)\s(?:de\s\d{1,2}\sde\s\w+(?:\sde)?\s\d{4}|\d+/\d+)", text)
    
    ministry_references = re.findall(r"(?:Ministerio|Consejería)(?:\sde\s\w+){1,4}(?:\sde\sfecha)?\s\d{1,2}\sde\s\w+(?:\sde)?\s\d{4}", text)
    
    official_bulletin = re.findall(r"(?:Boletín\sOficial|BOJA|BOCM|DOGC|BOPV|BOCYL)\s(?:del\s)?(?:Estado|Comunidad)?\s(?:núm\.|número)?\s\d+", text)
    
    all_references = []
    for ref in original_references:
        all_references.append(ref[0])
    all_references.extend(law_references)
    all_references.extend(ministry_references)
    all_references.extend(official_bulletin)
    
    return list(set(all_references))

def extract_submission_urls(text: str):
    # Original pattern for URLs
    urls = re.findall(r"https?://\S+", text)
    
    # New pattern for web platforms without full URL
    web_platforms = re.findall(r"(?:sede\selectrónica|portal\sweb|plataforma\sdigital|aplicación\sweb|sistema\sde\ssolicitud)\s(?:en|\:)?\s(?:la\spágina\sweb\s)?(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?)", text, flags=re.IGNORECASE)
    
    # Combine original URLs with web platforms
    all_urls = urls
    if web_platforms:
        for platform in web_platforms:
            if not any(platform in url for url in urls):
                # Add the platform as a potential URL if not already part of an existing URL
                if not platform.startswith(('http://', 'https://')):
                    platform = 'https://' + platform
                all_urls.append(platform)
    
    return list(set(all_urls))

def extract_eligibility_criteria(text: str):
    """Extract eligibility criteria related to nationality, residency, etc."""
    nationality = re.findall(r"(?:nacionalidad|ciudadanía)(?:\s\w+){1,4}(?:español(?:a)?|europea?|extranjero)", text, flags=re.IGNORECASE)
    
    residency = re.findall(r"(?:residencia|residir|empadronamiento)(?:\slegal)?(?:\sen\s\w+){1,3}(?:durante|mínimo|al\smenos|por\sun\speriodo)(?:\s\w+){0,3}\d+(?:\saños?|\smeses?)", text, flags=re.IGNORECASE)
    
    age_limits = re.findall(r"(?:edad|ser\smayor\sde|tener\sentre|menores\sde)(?:\s\w+){0,2}\s\d+(?:\s\w+){0,2}(?:años|edad)", text, flags=re.IGNORECASE)
    
    return {
        "nacionalidad": list(set(nationality)),
        "residencia": list(set(residency)),
        "límites_de_edad": list(set(age_limits))
    }

def extract_required_documents(text: str):
    """Extract information about required documentation."""
    document_keywords = [
        "dni", "nie", "pasaporte", "certificado", "expediente académico", 
        "declaración", "renta", "empadronamiento", "formulario", "solicitud",
        "título", "acreditación", "justificante"
    ]
    
    documents_pattern = r"(?:presentar|aportar|adjuntar|documentación|documentos)(?:\s\w+){0,5}(?:" + "|".join(document_keywords) + r")(?:\s\w+){0,15}(?:\.|,)"
    required_docs = re.findall(documents_pattern, text, flags=re.IGNORECASE)
    
    return list(set(required_docs))

def extract_incompatibilities(text: str):
    """Extract information about incompatibilities with other scholarships."""
    incompatibilities = re.findall(r"(?:incompatible|no\spodrá\sser\sbeneficiario|no\spodrán\ssolicitarla)(?:\s\w+){0,15}(?:beca|ayuda|subvención)(?:\s\w+){0,30}(?:\.|,)", text, flags=re.IGNORECASE)
    
    return list(set(incompatibilities))


# ---------- ESTRUCTURAR INFORMACIÓN ----------

def structure_information(text: str, entities: list, doc_name: str) -> dict:
    education_levels = extract_education_levels(text)
    scholarship_amounts = extract_amounts(text)
    income_thresholds = [i[1] for i in extract_income_thresholds(text)]
    academic_requirements = extract_academic_requirements(text)
    application_deadline = extract_deadlines(text)
    legal_references = extract_legal_references(text)
    submission_platform = extract_submission_urls(text)
    
    # New extractions
    eligibility = extract_eligibility_criteria(text)
    required_documents = extract_required_documents(text)
    incompatibilities = extract_incompatibilities(text)

    info = {
        "documento": doc_name,
        "autoridad_emisora": "",
        "niveles_educativos": sorted(set(education_levels)),
        "importes_beca": sorted(set(scholarship_amounts)),
        "umbrales_de_ingresos": sorted(set(income_thresholds)),
        "requisitos_académicos": sorted(set(academic_requirements)),
        "fecha_limite": sorted(set(application_deadline)),
        "referencias_legales": sorted(set(legal_references)),
        "plataforma": sorted(set(submission_platform)),
        # New fields
        "criterios_elegibilidad": eligibility,
        "documentacion_requerida": sorted(set(required_documents)),
        "incompatibilidades": sorted(set(incompatibilities))
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
    Nacionalidad/Residencia: {", ".join(data['criterios_elegibilidad']['nacionalidad'] + data['criterios_elegibilidad']['residencia'])}
    Edad: {", ".join(data['criterios_elegibilidad']['límites_de_edad'])}
    Documentación: {", ".join(data['documentacion_requerida'][:3])} y otros documentos
    Incompatibilidades: {", ".join(data['incompatibilidades'][:2])}
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