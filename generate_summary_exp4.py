import os
import json
from transformers import pipeline

# Carpetas de entrada y salida
INPUT_FOLDER = "data/parsed"
OUTPUT_FOLDER = "data/summaries_experiment4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_narrative_summary(parsed: dict) -> str:
    """
    Genera un resumen narrativo determinista a partir del JSON.
    Extrae los campos relevantes y los organiza en un párrafo, 
    reemplazando guiones bajos por espacios y separando las oraciones con puntos.
    """
    fecha_limite = parsed.get("plazo", {}).get("plazo_presentacion_fin")
    requisitos_minimos = parsed.get("requisitos", {}).get("matriculacion_minima", {})
    porcentajes = parsed.get("requisitos", {}).get("porcentajes_por_rama", {})
    cuantias = parsed.get("cuantias", {})
    excelencia = parsed.get("excelencia", {})
    presentacion = parsed.get("solicitud", {}).get("donde_presentar")
    
    partes = []
    
    if fecha_limite:
        partes.append(f"La convocatoria de beca establece que la fecha límite para presentar la solicitud es el {fecha_limite}.")
    else:
        partes.append("La convocatoria no especifica una fecha límite para la presentación de la solicitud.")
    
    if requisitos_minimos:
        req_list = [f"{k.replace('_', ' ').title()} requiere {v}" for k, v in requisitos_minimos.items()]
        req_text = ". ".join(req_list)
        partes.append(f"Entre los requisitos se exige: {req_text}.")
    
    if porcentajes:
        porc_list = [f"{rama.replace('_', ' ').title()} con un mínimo de {porc}%" for rama, porc in porcentajes.items()]
        porc_text = ". ".join(porc_list)
        partes.append(f"Asimismo, se establecen porcentajes mínimos por rama: {porc_text}.")
    
    if cuantias:
        cuant_list = [f"{clave.replace('_', ' ').title()} de {valor}" for clave, valor in cuantias.items()]
        cuant_text = ". ".join(cuant_list)
        partes.append(f"En términos económicos, la convocatoria dispone cuantías tales como {cuant_text}.")
    
    if excelencia:
        ex_list = [f"{clave.replace('_', ' ').title()} de {valor}" for clave, valor in excelencia.items()]
        ex_text = ". ".join(ex_list)
        partes.append(f"Además, se contemplan incentivos de excelencia, por ejemplo, {ex_text}.")
    
    if presentacion:
        partes.append(f"Los interesados deberán presentar la solicitud en: {presentacion}.")
    else:
        partes.append("No se especifica el lugar para presentar la solicitud.")
    
    summary_text = " ".join(partes).strip()
    if summary_text.endswith(".."):
        summary_text = summary_text[:-1]
    return summary_text

def process_all():
    # Inicializar los pipelines:
    # 1. Traducción de español a inglés.
    translator_es_en = pipeline(
        "translation_es_to_en",
        model="Helsinki-NLP/opus-mt-es-en",
        device=0  # Cambiar a -1 si no se tiene GPU
    )
    # 2. Resumen en inglés.
    english_summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        max_length=1024,
        min_length=200,
        truncation=True,
        device=-1
    )
    # 3. Traducción de inglés a español.
    translator_en_es = pipeline(
        "translation_en_to_es",
        model="Helsinki-NLP/opus-mt-en-es",
        device=-1
    )
    
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".json"):
            continue
        
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".json", "_resumen.txt"))
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            print(f"[DEBUG] {filename}: JSON cargado exitosamente.")
        except Exception as e:
            print(f"[ERROR] {filename}: Error al cargar el JSON: {e}")
            continue
        
        # Paso 1: Generar resumen narrativo determinista en español.
        narrative_summary = generate_narrative_summary(parsed)
        print(f"[DEBUG] {filename}: Resumen narrativo generado:\n{narrative_summary}\n{'-'*40}")
        
        # Paso 2: Traducir el resumen narrativo al inglés.
        try:
            english_translation = translator_es_en(narrative_summary)[0]["translation_text"].strip()
            print(f"[DEBUG] {filename}: Traducción al inglés:\n{english_translation}\n{'-'*40}")
        except Exception as e:
            print(f"[ERROR] {filename}: Error en la traducción al inglés: {e}")
            english_translation = narrative_summary  # fallback
        
        # Paso 3: Resumir el texto en inglés.
        try:
            summarized_english = english_summarizer(english_translation)[0]["summary_text"].strip()
            print(f"[DEBUG] {filename}: Resumen en inglés:\n{summarized_english}\n{'-'*40}")
        except Exception as e:
            print(f"[ERROR] {filename}: Error en el resumen en inglés: {e}")
            summarized_english = english_translation  # fallback
        
        # Paso 4: Traducir el resumen en inglés de vuelta al español.
        try:
            final_summary = translator_en_es(summarized_english)[0]["translation_text"].strip()
            print(f"[DEBUG] {filename}: Resumen final traducido al español:\n{final_summary}\n{'-'*40}")
        except Exception as e:
            print(f"[ERROR] {filename}: Error en la traducción de inglés a español: {e}")
            final_summary = summarized_english  # fallback
        
        # Guardar el resumen final en la carpeta de salida.
        try:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(final_summary)
            print(f"[INFO] {filename}: Resumen guardado en: {output_path}\n{'='*50}\n")
        except Exception as e:
            print(f"[ERROR] {filename}: Error al guardar el resumen: {e}")

if __name__ == "__main__":
    process_all()
