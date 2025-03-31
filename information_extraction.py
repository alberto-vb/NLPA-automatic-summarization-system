import pdfplumber
import re
import json
import os
import datetime
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from textwrap import wrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for more advanced PDF handling
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scholarship_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scholarship_extractor")

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------- ADVANCED CONFIGURATION ----------

# Regular expressions for common scholarship fields
REGEX_PATTERNS = {
    "amounts": r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?€",
    "deadlines": r"(?:hasta|antes del|plazo[^.]*?)\s+(?:el|día|fecha)?\s+\d{1,2}\s+de\s+\w+\s+(?:de|del)?\s+\d{4}",
    "income_thresholds": r"(renta.*?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?€)",
    "academic_requirements": r"(haber\s.+?créditos|nota\smedia\s.+?\d,\d+|superar\s.+?%)",
    "legal_references": r"(Real Decreto\s\d+/\d+|BOE\s(núm\.|\d+).+?\d{4})",
    "submission_urls": r"https?://\S+",
    "document_number": r"(?:Resolución|Convocatoria|Orden).+?(?:número|núm\.|nº)\s+([A-Z0-9-]+)",
    "phone_numbers": r"\d{3}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}",
    "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "student_types": r"(estudiantes\s+(?:de|universitarios|internacionales|extranjeros|con\s+discapacidad|de\s+máster|de\s+doctorado|de\s+grado))",
    # Add patterns for named entity recognition to replace spaCy
    "organizations": r"(?:Universidad|Ministerio|Fundación|Consejería|Instituto|Gobierno|Ayuntamiento|Diputación|Agencia)(?:\s+de|\s+del|\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+",
    "people": r"(?:Don|Doña|Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+",
    "locations": r"(?:en|de)\s+(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
    "dates": r"\d{1,2}\s+de\s+[a-záéíóúñ]+\s+(?:de|\s+)\d{4}",
    "money": r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:euros|€)"
}

# Predefined queries for QA extraction
QA_QUERIES = [
    "¿Cuál es el plazo de solicitud?",
    "¿Cuáles son los requisitos académicos?",
    "¿Qué cantidad económica ofrece la beca?",
    "¿Quién puede solicitar esta beca?",
    "¿Qué documentación se necesita presentar?",
    "¿Dónde se presenta la solicitud?",
    "¿Cuáles son los criterios de selección?",
    "¿Cuál es el umbral de renta máximo?",
    "¿Cuándo se publicará la resolución?"
]

# Categories of educational levels and their keywords
EDUCATION_LEVELS = {
    "educación_secundaria": ["educación secundaria", "eso", "secundaria", "educación secundaria obligatoria"],
    "bachillerato": ["bachillerato", "bachiller"],
    "formación_profesional": ["formación profesional", "fp", "ciclos formativos", "grado medio", "grado superior", "cfgm", "cfgs"],
    "grado": ["grado", "graduado", "título universitario", "estudios universitarios", "enseñanzas universitarias"],
    "máster": ["máster", "master", "posgrado", "postgrado"],
    "doctorado": ["doctorado", "phd", "doctor", "doctoral"]
}

# ---------- EXTRACCIÓN DE TEXTO AVANZADA ----------

def extract_text_from_pdf(pdf_path: str, method="pdfplumber") -> dict:
    """
    Extracts text from PDF using multiple methods and combines results.
    
    Args:
        pdf_path: Path to the PDF file
        method: The extraction method to use ("pdfplumber", "pymupdf", or "combined")
        
    Returns:
        Dictionary with extracted text by page and full text
    """
    logger.info(f"Extracting text from {pdf_path} using {method}")
    result = {"pages": {}, "full_text": ""}
    
    try:
        if method == "pdfplumber" or method == "combined":
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if method == "combined":
                        result["pages"][f"page_{i}_pdfplumber"] = page_text
                    else:
                        result["pages"][f"page_{i}"] = page_text
                    result["full_text"] += page_text + "\n"
                
        if method == "pymupdf" or method == "combined":
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc, 1):
                    page_text = page.get_text() or ""
                    if method == "combined":
                        result["pages"][f"page_{i}_pymupdf"] = page_text
                        # If using combined, only add PyMuPDF text if it adds significant content
                        pdfplumber_text = result["pages"].get(f"page_{i}_pdfplumber", "")
                        if len(page_text) > len(pdfplumber_text) * 1.2:  # 20% more text
                            result["full_text"] += page_text + "\n"
                    else:
                        result["pages"][f"page_{i}"] = page_text
                        result["full_text"] += page_text + "\n"
        
        # Extract tables from PDF as well
        tables = extract_tables_from_pdf(pdf_path)
        if tables:
            result["tables"] = tables
            
        return result
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return {"pages": {}, "full_text": "", "error": str(e)}

def extract_tables_from_pdf(pdf_path: str) -> list:
    """Extract tables from PDF and convert to structured data"""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for j, table in enumerate(page_tables, 1):
                    if table:
                        # Convert to pandas DataFrame for easier manipulation
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        tables.append({
                            "page": i,
                            "table_index": j,
                            "data": df.to_dict(orient="records")
                        })
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables from {pdf_path}: {e}")
        return []

def extract_metadata_from_pdf(pdf_path: str) -> dict:
    """Extract PDF metadata like author, creation date, title"""
    metadata = {}
    try:
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            # Add page count
            metadata["page_count"] = len(doc)
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        return {}

# ---------- PROCESAMIENTO DE TEXTO AVANZADO ----------

def preprocess_text(text: str) -> str:
    """Clean and normalize text for better extraction"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('–', '-')
    # Remove headers/footers that match common patterns
    text = re.sub(r'Página \d+ de \d+', '', text)
    text = re.sub(r'(?i)código seguro de verificación.+?firmante.*?\d{2}:\d{2}', '', text)
    return text.strip()

def segment_text(text: str) -> dict:
    """
    Segment text into sections based on common scholarship document structure
    """
    sections = {
        "introduction": "",
        "requirements": "",
        "documentation": "",
        "procedure": "",
        "deadlines": "",
        "selection_criteria": "",
        "benefits": "",
        "obligations": "",
        "other": ""
    }
    
    sentences = sent_tokenize(text)
    
    # Dictionary of keywords that suggest a particular section
    section_keywords = {
        "introduction": ["objeto", "finalidad", "introducción", "convoca", "convocatoria"],
        "requirements": ["requisitos", "podrán solicitar", "podrán participar", "destinatarios"],
        "documentation": ["documentación", "documentos", "acreditar", "presentar", "solicitud"],
        "procedure": ["procedimiento", "tramitación", "forma de presentación", "solicitudes"],
        "deadlines": ["plazo", "fecha límite", "hasta el día", "término"],
        "selection_criteria": ["criterios de selección", "baremo", "valoración", "evaluación"],
        "benefits": ["dotación", "cuantía", "importe", "euros", "€", "cantidad"],
        "obligations": ["obligaciones", "compromiso", "deber"],
    }
    
    # Assign sentences to sections based on keyword matching
    for sentence in sentences:
        sentence_lower = sentence.lower()
        assigned = False
        
        for section, keywords in section_keywords.items():
            if any(keyword in sentence_lower for keyword in keywords):
                sections[section] += sentence + " "
                assigned = True
                break
                
        if not assigned:
            sections["other"] += sentence + " "
    
    return {k: v.strip() for k, v in sections.items() if v.strip()}

# ---------- NAMED ENTITY RECOGNITION REPLACEMENT ----------

def extract_entities(text: str, chunk_size: int = 400) -> dict:
    """
    Extract named entities using regex patterns instead of spaCy
    Groups entities by type.
    """
    # Initialize results dictionary
    entities_by_type = {
        "ORG": [],  # Organizations
        "PER": [],  # People
        "LOC": [],  # Locations
        "DATE": [], # Dates
        "MONEY": [] # Monetary values
    }
    
    # Use regex patterns to extract entities
    org_matches = set(re.findall(REGEX_PATTERNS["organizations"], text, re.IGNORECASE))
    per_matches = set(re.findall(REGEX_PATTERNS["people"], text, re.IGNORECASE))
    loc_matches = set(re.findall(REGEX_PATTERNS["locations"], text, re.IGNORECASE))
    date_matches = set(re.findall(REGEX_PATTERNS["dates"], text, re.IGNORECASE))
    money_matches = set(re.findall(REGEX_PATTERNS["money"], text, re.IGNORECASE))
    
    # Clean up extracted entities
    entities_by_type["ORG"] = sorted([m.strip() for m in org_matches if len(m.strip()) > 3])
    entities_by_type["PER"] = sorted([m.strip() for m in per_matches if len(m.strip()) > 3])
    entities_by_type["LOC"] = sorted([m.strip() for m in loc_matches if len(m.strip()) > 3])
    entities_by_type["DATE"] = sorted([m.strip() for m in date_matches if len(m.strip()) > 3])
    entities_by_type["MONEY"] = sorted([m.strip() for m in money_matches if len(m.strip()) > 3])
    
    # Optional: use transformers NER for better accuracy if available
    try:
        ner_pipeline = pipeline(
            "ner",
            model="mrm8488/bert-spanish-cased-finetuned-ner",
            aggregation_strategy="simple"
        )
        
        chunks = wrap(text, chunk_size)
        
        for chunk in chunks:
            try:
                transformer_entities = ner_pipeline(chunk)
                for ent in transformer_entities:
                    if ent["entity_group"] in ["ORG", "PER", "LOC"]:
                        mapped_type = {
                            "ORG": "ORG", 
                            "PER": "PER", 
                            "LOC": "LOC"
                        }.get(ent["entity_group"])
                        
                        if mapped_type and ent["word"] not in entities_by_type[mapped_type]:
                            entities_by_type[mapped_type].append(ent["word"])
            except Exception as e:
                logger.warning(f"Error in Transformers NER chunk processing: {e}")
    except Exception as e:
        logger.warning(f"Transformers NER not available, using regex only: {e}")
    
    # Clean up results and remove duplicates
    for entity_type in entities_by_type:
        entities_by_type[entity_type] = sorted(list(set(entities_by_type[entity_type])))
    
    return entities_by_type

# ---------- EXTRACCIÓN BASADA EN QA ----------

def extract_info_with_qa(text: str, queries: list = None) -> dict:
    """
    Use Question-Answering model to extract specific information
    """
    if queries is None:
        queries = QA_QUERIES
        
    try:
        # Load QA model for Spanish
        tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne-sqac")
        model = AutoModelForQuestionAnswering.from_pretrained("PlanTL-GOB-ES/roberta-base-bne-sqac")
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        
        results = {}
        
        # Process text in chunks if it's too long
        max_length = 512
        text_chunks = []
        
        if len(text) > max_length:
            # Split text into sentences and group them into chunks
            sentences = sent_tokenize(text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + " "
                else:
                    text_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            # Add the last chunk
            if current_chunk:
                text_chunks.append(current_chunk.strip())
        else:
            text_chunks = [text]
        
        # Run QA on each query and each chunk
        for query in queries:
            best_answer = {"score": 0, "answer": ""}
            
            for chunk in text_chunks:
                try:
                    result = qa_pipeline(question=query, context=chunk)
                    
                    # Keep the best answer based on score
                    if result["score"] > best_answer["score"]:
                        best_answer = result
                except Exception as e:
                    logger.error(f"Error in QA for query '{query}': {e}")
            
            if best_answer["score"] > 0.2:  # Only keep reasonably confident answers
                # Clean up and format the key
                key = query.lower()
                key = re.sub(r'¿|cuál es |cuáles son |qué |dónde |cuándo |cómo |la |el |los |las |se |una |un ', '', key)
                key = key.replace(' ', '_').replace('?', '').strip()
                
                results[key] = best_answer["answer"]
        
        return results
    
    except Exception as e:
        logger.error(f"Error setting up QA pipeline: {e}")
        return {}

# ---------- EXTRACCIÓN REGLADA DE CAMPOS MEJORADA ----------

def extract_field_with_regex(text: str, pattern_name: str) -> list:
    """Extract information using predefined regex patterns"""
    pattern = REGEX_PATTERNS.get(pattern_name)
    if not pattern:
        logger.warning(f"No pattern defined for {pattern_name}")
        return []
    
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    
    # Handle tuples returned by regex groups
    if matches and isinstance(matches[0], tuple):
        # For cases where we want the whole match and a specific group
        if pattern_name == "income_thresholds":
            return [m[1] for m in matches]  # Return just the amounts
        return [m[0] for m in matches]  # Return the full match
    
    return matches

def extract_education_levels(text: str) -> dict:
    """
    Extract education levels with more detailed categorization
    """
    text_lower = text.lower()
    found_levels = {}
    
    for category, keywords in EDUCATION_LEVELS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Find the full context around this keyword
                # Look for sentences containing the keyword
                matches = re.findall(r'[^.!?]*(?<=[.!?\s])' + re.escape(keyword) + r'(?=[.!?\s])[^.!?]*[.!?]', text_lower)
                
                if matches:
                    found_levels[category] = {
                        "detected": True,
                        "keyword": keyword,
                        "context": [m.strip() for m in matches]
                    }
                    break
                else:
                    found_levels[category] = {
                        "detected": True,
                        "keyword": keyword,
                        "context": []
                    }
                    break
        
        if category not in found_levels:
            found_levels[category] = {"detected": False}
    
    return found_levels

def extract_document_date(text: str) -> str:
    """Extract the document publication/creation date"""
    # Look for common date patterns in Spanish documents
    date_patterns = [
        r'(?:Madrid|Barcelona|Valencia|Sevilla),?\s+a\s+(\d{1,2})\s+de\s+([a-zñáéíóú]+)\s+de\s+(\d{4})',
        r'(?:fecha|publicado el|con fecha)\s+(?:de|del)?\s*?(\d{1,2})\s+de\s+([a-zñáéíóú]+)\s+(?:de|del)?\s+(\d{4})',
        r'(\d{1,2})\s+de\s+([a-zñáéíóú]+)\s+de\s+(\d{4})'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            day, month, year = matches[0]
            
            # Normalize Spanish month names
            month_map = {
                'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
                'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
                'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
            }
            
            month_num = month_map.get(month.lower(), '00')
            return f"{day.zfill(2)}/{month_num}/{year}"
    
    return ""

def extract_contact_information(text: str) -> dict:
    """Extract contact information like phones, emails, websites"""
    contact_info = {
        "phones": extract_field_with_regex(text, "phone_numbers"),
        "emails": extract_field_with_regex(text, "emails"),
        "websites": extract_field_with_regex(text, "submission_urls")
    }
    return contact_info

# ---------- SEMANTIC SEARCH AND CLASSIFICATION ----------

def create_semantic_index(documents: list) -> tuple:
    """Create a semantic search index for document comparison"""
    vectorizer = TfidfVectorizer(
        min_df=1, 
        ngram_range=(1, 2),
        stop_words=['de', 'la', 'el', 'y', 'en', 'a', 'que', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con']
    )
    
    # Extract just the text content and document names
    texts = [doc["full_text"] for doc in documents]
    doc_names = [doc["file_name"] for doc in documents]
    
    # Create the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return vectorizer, tfidf_matrix, doc_names

def find_similar_documents(query_text: str, vectorizer, tfidf_matrix, doc_names, top_n=3) -> list:
    """Find documents similar to a query text"""
    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([query_text])
    
    # Calculate similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top N similar documents
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Return document names and scores
    results = [
        {"document": doc_names[i], "similarity": similarity_scores[i]} 
        for i in top_indices if similarity_scores[i] > 0
    ]
    
    return results

def classify_scholarship_type(text: str) -> list:
    """
    Classify the scholarship by type based on text content.
    Returns a list of probable scholarship types with confidence scores.
    """
    # Define keyword sets for different scholarship types
    scholarship_types = {
        "mobility": ["movilidad", "erasmus", "internacional", "intercambio", "extranjero"],
        "research": ["investigación", "doctorado", "proyecto", "laboratorio", "tesis"],
        "general": ["general", "ordinaria", "matrícula", "tasas", "curso académico"],
        "excellence": ["excelencia", "rendimiento", "alto rendimiento", "premios", "extraordinario"],
        "need_based": ["necesidad", "situación económica", "emergencia", "vulnerabilidad", "renta"],
        "disability": ["discapacidad", "diversidad funcional", "necesidades especiales"],
        "postgraduate": ["posgrado", "postgrado", "máster", "especialización"],
        "internship": ["prácticas", "empresa", "profesional", "formación práctica"]
    }
    
    results = []
    text_lower = text.lower()
    
    for s_type, keywords in scholarship_types.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        if count > 0:
            confidence = min(count / len(keywords) * 2, 1.0)  # Normalize to max 1.0
            results.append({"type": s_type, "confidence": confidence})
    
    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return results

# ---------- STRUCTURING INFORMATION ----------

def structure_information(text: str, extraction_results: dict, doc_name: str) -> dict:
    """
    Combine all extraction methods into a structured output
    """
    # Clean and preprocess text
    clean_text = preprocess_text(text)
    
    # Extract basic information using regex
    education_levels = extract_education_levels(clean_text)
    scholarship_amounts = extract_field_with_regex(clean_text, "amounts")
    income_thresholds = extract_field_with_regex(clean_text, "income_thresholds")
    academic_requirements = extract_field_with_regex(clean_text, "academic_requirements")
    application_deadline = extract_field_with_regex(clean_text, "deadlines")
    legal_references = extract_field_with_regex(clean_text, "legal_references")
    submission_platform = extract_field_with_regex(clean_text, "submission_urls")
    
    # Get entities from extraction_results
    entities = extraction_results.get("entities", {})
    
    # Get QA results if available
    qa_results = extraction_results.get("qa_results", {})
    
    # Get document metadata if available
    metadata = extraction_results.get("metadata", {})
    
    # Extract document date
    document_date = extract_document_date(clean_text)
    
    # Extract contact information
    contact_info = extract_contact_information(clean_text)
    
    # Get document segmentation
    sections = segment_text(clean_text)
    
    # Classify scholarship type
    scholarship_types = classify_scholarship_type(clean_text)
    
    # Construct the structured information
    info = {
        "documento": {
            "nombre": doc_name,
            "fecha": document_date,
            "metadata": metadata
        },
        "autoridad_emisora": "",
        "tipo_beca": scholarship_types,
        "niveles_educativos": education_levels,
        "importes_beca": sorted(set(scholarship_amounts)),
        "umbrales_de_ingresos": sorted(set(income_thresholds)),
        "requisitos_académicos": sorted(set(academic_requirements)),
        "fecha_limite": sorted(set(application_deadline)),
        "referencias_legales": sorted(set(legal_references)),
        "plataforma": sorted(set(submission_platform)),
        "contacto": contact_info,
        "secciones": sections,
        "qa_respuestas": qa_results
    }
    
    # Set the issuing authority if found
    organizations = entities.get("ORG", [])
    if organizations:
        info["autoridad_emisora"] = organizations[0]
    
    # Remove empty fields for cleaner output
    info = {k: v for k, v in info.items() if v}
    
    return info

# ---------- GENERACIÓN DE RESUMEN MEJORADO ----------

def generate_summary(data: dict, model="google/flan-t5-large") -> dict:
    """
    Generate a comprehensive summary in multiple formats:
    - Short summary (1-2 sentences)
    - Bullet points for key details
    - Full narrative summary
    """
    try:
        summarizer = pipeline("text2text-generation", model=model, max_length=512)
        
        # Extract relevant data for the summary
        doc_name = data.get("documento", {}).get("nombre", "documento")
        issuer = data.get("autoridad_emisora", "")
        education = []
        for level, details in data.get("niveles_educativos", {}).items():
            if details.get("detected"):
                education.append(level.replace("_", " "))
        
        amounts = data.get("importes_beca", [])
        income = data.get("umbrales_de_ingresos", [])
        requirements = data.get("requisitos_académicos", [])
        deadline = data.get("fecha_limite", [])
        platform = data.get("plataforma", [])
        
        # Create prompt for the short summary
        short_prompt = f"""
        Resume en una frase esta beca: {doc_name} de {issuer} para estudiantes de {', '.join(education)[:100]}.
        Importe: {', '.join(amounts)[:50]}. Fecha límite: {', '.join(deadline)[:50]}.
        """
        
        # Create prompt for the full summary
        full_prompt = f"""
        Genera un resumen completo y claro para estudiantes sobre la beca publicada en {doc_name}.

        Organismo: {issuer}
        Niveles educativos: {', '.join(education)}
        Cuantía: {', '.join(amounts)}
        Umbral de renta: {', '.join(income)}
        Requisitos académicos: {', '.join(requirements)}
        Plazo de solicitud: {', '.join(deadline)}
        Más info: {', '.join(platform)}
        """
        
        # Generate summaries
        short_summary = summarizer(short_prompt)[0]["generated_text"]
        full_summary = summarizer(full_prompt)[0]["generated_text"]
        
        # Create bullet points from key data
        bullet_points = []
        if issuer:
            bullet_points.append(f"Convoca: {issuer}")
        if education:
            bullet_points.append(f"Para estudiantes de: {', '.join(education)}")
        if amounts:
            bullet_points.append(f"Importe: {', '.join(amounts[:3])}")
        if deadline:
            bullet_points.append(f"Plazo: {', '.join(deadline[:1])}")
        if requirements:
            bullet_points.append(f"Requisitos clave: {', '.join(requirements[:2])}")
        if platform:
            bullet_points.append(f"Solicitud: {', '.join(platform[:1])}")
        
        return {
            "short_summary": short_summary,
            "bullet_points": bullet_points,
            "full_summary": full_summary
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {
            "short_summary": f"Beca publicada en {doc_name}.",
            "bullet_points": ["Error al generar puntos clave."],
            "full_summary": f"Error al generar resumen completo para {doc_name}."
        }

# ---------- TEMPORAL ANALYSIS ----------

def compare_scholarship_years(scholarship_data: list) -> dict:
    """
    Compare scholarship details across different years to identify changes
    """
    if len(scholarship_data) < 2:
        return {"error": "Need at least two scholarship entries to compare"}
    
    # Sort data by date if available
    try:
        sorted_data = sorted(scholarship_data, 
                            key=lambda x: x.get("documento", {}).get("fecha", ""), 
                            reverse=True)
    except:
        sorted_data = scholarship_data
    
    comparison = {
        "scholarships_compared": len(sorted_data),
        "changes": {
            "importes_beca": {},
            "umbrales_de_ingresos": {},
            "requisitos_académicos": {},
            "fecha_limite": {},
            "plataforma": {}
        },
        "temporal_trends": {},
        "summary": ""
    }
    
    # Track changes across all documents
    for i in range(len(sorted_data) - 1):
        current = sorted_data[i]
        previous = sorted_data[i + 1]
        
        current_year = current.get("documento", {}).get("fecha", "")[-4:]
        previous_year = previous.get("documento", {}).get("fecha", "")[-4:]
        
        if not current_year or not previous_year:
            continue
            
        # Compare key fields
        for field in ["importes_beca", "umbrales_de_ingresos", "requisitos_académicos", 
                      "fecha_limite", "plataforma"]:
            current_values = set(current.get(field, []))
            previous_values = set(previous.get(field, []))
            
            added = current_values - previous_values
            removed = previous_values - current_values
            
            if added or removed:
                if f"{previous_year}_to_{current_year}" not in comparison["changes"][field]:
                    comparison["changes"][field][f"{previous_year}_to_{current_year}"] = {
                        "added": list(added),
                        "removed": list(removed)
                    }
    
    # Analyze trends in scholarship amounts
    all_amounts = []
    for data in sorted_data:
        doc_year = data.get("documento", {}).get("fecha", "")[-4:]
        if not doc_year:
            continue
            
        for amount_str in data.get("importes_beca", []):
            # Extract numeric values from amount strings
            match = re.search(r'(\d+(?:\.\d+)*(?:,\d+)?)', amount_str)
            if match:
                amount_num = float(match.group(1).replace('.', '').replace(',', '.'))
                all_amounts.append((doc_year, amount_num))
    
    # Group amounts by year
    amounts_by_year = {}
    for year, amount in all_amounts:
        if year not in amounts_by_year:
            amounts_by_year[year] = []
        amounts_by_year[year].append(amount)
    
    # Calculate average amount per year
    avg_by_year = {year: sum(amounts)/len(amounts) for year, amounts in amounts_by_year.items()}
    
    # Calculate percentage change between years
    years = sorted(avg_by_year.keys())
    for i in range(len(years) - 1):
        current_year = years[i+1]
        prev_year = years[i]
        
        if avg_by_year[prev_year] > 0:
            percent_change = ((avg_by_year[current_year] - avg_by_year[prev_year]) / 
                             avg_by_year[prev_year] * 100)
            
            comparison["temporal_trends"][f"{prev_year}_to_{current_year}"] = {
                "avg_amount_prev": avg_by_year[prev_year],
                "avg_amount_curr": avg_by_year[current_year],
                "percent_change": round(percent_change, 2)
            }
    
    # Generate a summary of the changes
    changes_detected = any(bool(changes) for changes in comparison["changes"].values())
    
    if changes_detected:
        summary = "Se han detectado cambios significativos entre convocatorias: "
        
        for field, years_changes in comparison["changes"].items():
            if years_changes:
                field_name = {
                    "importes_beca": "importes",
                    "umbrales_de_ingresos": "umbrales de renta",
                    "requisitos_académicos": "requisitos académicos",
                    "fecha_limite": "plazos de solicitud",
                    "plataforma": "plataformas de solicitud"
                }.get(field, field)
                
                summary += f"cambios en {field_name}, "
        
        # Add trend information
        if comparison["temporal_trends"]:
            trend_info = []
            for year_range, trend in comparison["temporal_trends"].items():
                if abs(trend["percent_change"]) > 5:  # Only report significant changes
                    direction = "aumentado" if trend["percent_change"] > 0 else "disminuido"
                    trend_info.append(
                        f"los importes han {direction} un {abs(trend['percent_change']):.1f}% " 
                        f"de {year_range.split('_to_')[0]} a {year_range.split('_to_')[1]}"
                    )
            
            if trend_info:
                summary += "con tendencias notables: " + "; ".join(trend_info)
        
        comparison["summary"] = summary.rstrip(", ")
    else:
        comparison["summary"] = "No se han detectado cambios significativos entre las convocatorias analizadas."
    
    return comparison