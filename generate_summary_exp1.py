import os
import json
from transformers import pipeline

# Carpetas de entrada (JSON) y de salida (resúmenes)
INPUT_FOLDER = "data/parsed"
OUTPUT_FOLDER = "data/summaries_experiment1"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_all():
    # Definimos un pipeline con un modelo T5 (por ejemplo, google/flan-t5-large)
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=1024,     # Ajustar si quieres más/menos longitud
        truncation=True,
        device=0           # GPU; cambiar a -1 si quieres usar CPU y está configurada
    )
    
    # Recorremos todos los JSON de la carpeta de entrada
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".json"):
            continue
        
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".json", "_resumen.txt"))
        
        # Cargar el JSON tal cual
        with open(input_path, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)
        
        # Convertir el contenido del JSON a una cadena (para incrustarlo en el prompt)
        json_text = json.dumps(parsed_data, indent=2, ensure_ascii=False)
        
        # Prompt: pasamos el JSON directamente y pedimos un resumen en español
        prompt = (
            "Eres un experto en la síntesis de datos en JSON. Por favor, elabora un resumen "
            "en ESPAÑOL sin reescribirlo textualmente ni enumerarlo. Sé claro, conciso y directo:\n\n"
            + json_text
        )
        
        try:
            # Generar el resumen con el pipeline
            ai_output = summarizer(prompt)
            ai_summary = ai_output[0]["generated_text"].strip()
            print(f"[DEBUG] {filename}: Resumen T5:\n{ai_summary}\n{'-'*40}")
        except Exception as e:
            print(f"[ERROR] {filename}: Error al generar el resumen: {e}")
            ai_summary = "Error generando el resumen."
        
        # Guardar el resultado
        try:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(ai_summary)
            print(f"[INFO] {filename}: Resumen guardado en {output_path}\n{'='*50}\n")
        except Exception as e:
            print(f"[ERROR] {filename}: Error al guardar el resumen: {e}")

if __name__ == "__main__":
    process_all()
