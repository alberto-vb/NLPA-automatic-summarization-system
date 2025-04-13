import os
import json
from transformers import pipeline
from tqdm import tqdm

# Solucionar primero la instalación de SentencePiece
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configuración de carpetas
INPUT_FOLDERS = {
    "resumenes_regex": "resumenes_txt/resumenes_regex",
    "resumenes_semantic": "resumenes_txt/resumenes_semantic",
    "resumenes_spacy": "resumenes_txt/resumenes_spacy"
}

# Inicializar el modelo de resumen (usamos pipeline para simplificar)
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def format_json_content(data):
    """Convierte el contenido JSON en texto plano para resumir"""
    text_parts = []
    
    if "fechas" in data and "montos" in data:  # Estructura de regex
        text_parts.append("Fechas relevantes:")
        text_parts.extend(data["fechas"])
        text_parts.append("\nMontos mencionados:")
        text_parts.extend(data["montos"])
    
    elif "convocatoria" in data:  # Estructura de semantic
        text_parts.append("Convocatoria:")
        text_parts.extend(data["convocatoria"])
        if "requisitos" in data:
            text_parts.append("\nRequisitos:")
            text_parts.extend(data["requisitos"])
        if "plazo" in data:
            text_parts.append("\nPlazos:")
            text_parts.extend(data["plazo"])
    
    elif "entidades" in data:  # Estructura de spacy
        for ent_type, entities in data["entidades"].items():
            if entities:
                text_parts.append(f"{ent_type}:")
                for ent in entities:
                    text_parts.append(f"- {ent['frase']}")
        if "frases_accion" in data:
            text_parts.append("\nFrases de acción:")
            for frase in data["frases_accion"]:
                text_parts.append(f"- {frase['frase']}")
    
    return "\n".join(text_parts)

def generate_summary(text):
    """Genera un resumen usando el modelo T5"""
    try:
        # Preprocesar texto para el modelo (limitar a 512 tokens)
        inputs = summarizer.tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        
        # Generar resumen
        summary_ids = summarizer.model.generate(
            input_ids,
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        return summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error al generar resumen: {str(e)}")
        return "No se pudo generar el resumen"

def process_files():
    """Procesa todos los archivos JSON y genera resúmenes TXT"""
    for input_folder, output_folder in INPUT_FOLDERS.items():
        # Crear carpeta de salida si no existe
        os.makedirs(output_folder, exist_ok=True)
        
        # Procesar cada archivo JSON
        for filename in tqdm(os.listdir(input_folder), desc=f"Procesando {input_folder}"):
            if filename.endswith(".json"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename.replace(".json", ".txt"))
                
                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Convertir JSON a texto para resumir
                    text_to_summarize = format_json_content(data)
                    
                    # Generar y guardar resumen
                    summary = generate_summary(text_to_summarize)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                
                except Exception as e:
                    print(f"Error procesando {filename}: {str(e)}")

if __name__ == "__main__":
    # Instalar dependencias si faltan
    try:
        process_files()
    except Exception as e:
        print(f"Error general: {str(e)}")
        print("Asegúrate de tener instaladas todas las dependencias:")
        print("pip install transformers sentencepiece tqdm")