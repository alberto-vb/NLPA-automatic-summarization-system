import pdfplumber
import re
import json
from transformers import pipeline
from textwrap import wrap
import os

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
    return re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?(?:€|euros|EUR)", text, flags=re.IGNORECASE)

def extract_deadlines(text: str):
    return re.findall(r"hasta el \d{1,2} de \w+ de \d{4}", text, flags=re.IGNORECASE)

def extract_income_thresholds(text: str):
    return re.findall(r"(?:renta|ingresos|familias).{0,80}?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?(?:€|euros|EUR)", text, flags=re.IGNORECASE)

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
    text = re.sub(r"(\.\s*){3,}", " ", text)

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

def limpiar_resumen(texto: str) -> str:
    # Quitar frases como "Genera un resumen..." y puntos intermedios
    texto = re.sub(r"(Genera(r)? un resumen.*?:|Resumen del documento.*?)", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"(\.\s*){3,}", " ", texto)  # puntos intermedios
    return texto.strip()

def construir_resumen_base(niveles, cuantias, umbrales, requisitos, plazos):
    texto = (
        f"Las becas BOE están destinadas a estudiantes de niveles como: {niveles}. "
        f"Las cuantías económicas ofrecidas suelen incluir: {cuantias}. "
        f"En cuanto a la situación económica, los umbrales de renta aplicables son: {umbrales}. "
        f"Los requisitos académicos más comunes incluyen: {requisitos}. "
        f"Los plazos habituales para solicitar estas becas suelen ser: {plazos}."
    )
    return texto

def construir_resumen_base(niveles, cuantias, umbrales, requisitos, plazos):
    texto = (
        f"Las becas BOE están destinadas a estudiantes de niveles como: {niveles}. "
        f"Las cuantías económicas ofrecidas suelen incluir: {cuantias}. "
        f"En cuanto a la situación económica, los umbrales de renta aplicables son: {umbrales}. "
        f"Los requisitos académicos más comunes incluyen: {requisitos}. "
        f"Los plazos habituales para solicitar estas becas suelen ser: {plazos}."
    )
    return texto

def resumir_importes(cuantias_raw):
    import re

    cantidades = []

    for valor in cuantias_raw:
        # Eliminar palabras y símbolos
        limpio = re.sub(r"[^\d,\.]", "", valor.lower())

        # Cambiar "1.600,00" → "1600.00"
        if "," in limpio and "." in limpio:
            # Asumimos formato español con punto como miles
            limpio = limpio.replace(".", "").replace(",", ".")
        elif "," in limpio:
            limpio = limpio.replace(",", ".")
        elif "." in limpio:
            parts = limpio.split(".")
            if len(parts[-1]) == 2:
                # Probablemente decimal (e.g., 1500.00)
                pass
            else:
                # Miles con punto → eliminar
                limpio = limpio.replace(".", "")

        try:
            num = float(limpio)
            if 10 <= num <= 10000:  # filtrar valores absurdos tipo 1.850.000.000
                cantidades.append(num)
        except ValueError:
            continue

    if not cantidades:
        return "no indicadas"

    minimo = min(cantidades)
    maximo = max(cantidades)

    return f"entre {minimo:,.2f} y {maximo:,.2f} euros".replace(",", "X").replace(".", ",").replace("X", ".")

def generar_metaresumen_directo(directorio_jsons: str = "corpus", modelo: str = "google/flan-t5-large") -> str:
    from transformers import pipeline
    import os, json

    generador = pipeline("text2text-generation", model=modelo)

    niveles_all = set()
    cuantias_all = set()
    umbrales_all = set()
    requisitos_all = set()
    plazos_all = set()

    for archivo in os.listdir(directorio_jsons):
        if archivo.endswith("_info.json"):
            ruta = os.path.join(directorio_jsons, archivo)
            with open(ruta, "r", encoding="utf-8") as f:
                beca = json.load(f)

            niveles_all.update(beca.get("niveles_educativos", []))
            cuantias_all.update(beca.get("importes_beca", []))
            umbrales_all.update(beca.get("umbrales_de_ingresos", []))
            requisitos_all.update(beca.get("requisitos_académicos", []))
            plazos_all.update(beca.get("fecha_limite", []))

    # 🧩 Aquí colocas estas líneas:
    niveles = ", ".join(sorted(niveles_all)) or "no especificados"
    cuantias = resumir_importes(cuantias_all)
    umbrales = ", ".join(sorted(umbrales_all)) or "no indicados"
    requisitos = ", ".join(sorted(requisitos_all)) or "no especificados"
    plazos = ", ".join(sorted(plazos_all)) or "no indicados"

    resumen_base = construir_resumen_base(niveles, cuantias, umbrales, requisitos, plazos)

    # Reformulación con modelo
    prompt = ("Reformula el siguiente texto de forma clara, concisa y natural para estudiantes:\n\n"
    f"{resumen_base}"
    )

    resultado = generador(prompt, max_length=512, truncation=True)[0]["generated_text"]
    return resultado

# Guardar en txt si se desea
def guardar_resumen_global(resumen: str, archivo: str = "resumen_global.txt"):
    with open(archivo, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DE BECAS BOE\n")
        f.write("=" * 50 + "\n\n")
        f.write(resumen)
    print(f"Resumen guardado en: {archivo}")
# ---------- MAIN ----------

if __name__ == "__main__":
    pdfs = [
        'corpus/ayudas_25-26.pdf',
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

        # Guardar JSON individual con la información estructurada
        json_path = f"corpus/{os.path.basename(pdf).replace('.pdf', '_info.json')}"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

    # Generar y guardar resumen global directamente desde los JSONs individuales
    resumen_global = generar_metaresumen_directo("corpus")
    print("\nResumen global generado:\n")
    print(resumen_global)

    guardar_resumen_global(resumen_global)