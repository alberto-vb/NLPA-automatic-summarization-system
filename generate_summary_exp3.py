import os
import json
from transformers import pipeline

# Carpetas de entrada y salida
INPUT_FOLDER = "data/parsed"
OUTPUT_FOLDER = "data/summaries_experiment3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_narrative_summary(parsed: dict) -> str:
    """
    Genera un resumen narrativo determinista a partir de los datos extraídos del JSON.
    Redacta uno o dos párrafos de forma natural, reemplazando guiones bajos por espacios
    y usando el punto (.) como separador de oraciones.
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

def refine_text(text: str) -> str:
    """
    Aplica una serie de correcciones sobre el texto para refinarlo,
    eliminando mezclas de idiomas y errores tipográficos, y corrigiendo frases
    que impliquen especial énfasis en la cuantía de la renta y en la fecha límite.
    """
    replacements = {
        "the solicitud": "la solicitud",
        "or denegación": "o denegación",
        "with un mminimo": "con un mínimo",
        "mminimo": "mínimo",
        "mnimo": "mínimo",
        "ensanzas": "enseñanzas",
        "Ensanzas": "Enseñanzas",
        "Msica": "Música",
        "Diseo": "Diseño",
        "Capitulo": "CAPÍTULO",
        "with ": "con ",
        " of ": " de ",
        " un mminimo": " un mínimo",
        "mnimos": "mínimos",
        # Nuevas correcciones específicas:
        "Excel Valencia": "Excelencia",
        "BecaBasica Fp Basico": "Beca Basica FP Basico",
        "BecA Basica FP Basico": "Beca Basica FP Basico",
        "and": "y",
        "  ,": ",",        # quitar espacios dobles antes de comas
        " ,": ",",         # eliminar espacios antes de comas
        " ,": ",",
        " Fija Basica normal": " Beca Basica Normal",
        "Variables Minima": "Variable Minima",
        "Beca Basica Fp Basico de 60,000": "",  # Si aparece repetido o erróneo
    }
    refined = text
    for wrong, right in replacements.items():
        refined = refined.replace(wrong, right)
    # Opcionalmente, podríamos usar expresiones regulares para eliminar repeticiones o ajustar espacios.
    return refined

def process_all():
    # Inicializar el pipeline de resumen con BART.
    ai_summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        max_length=1024,
        min_length=300,
        truncation=True,
        device=0  # Cambia a -1 si no dispones de GPU
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
        
        narrative_summary = generate_narrative_summary(parsed)
        print(f"[DEBUG] {filename}: Resumen narrativo generado:\n{narrative_summary}\n{'-'*40}")
        
        # Ajustamos el prompt para enfatizar especialmente la cuantía ligada a la renta y la fecha límite.
        prompt = ("Reformula de forma natural y redactada el siguiente texto en español, "
                  "haciendo especial énfasis en la cuantía relacionada con la renta y en la fecha límite de presentación, "
                  "ya que son aspectos críticos:\n\n" + narrative_summary)
        try:
            ai_output = ai_summarizer(prompt)
            ai_summary = ai_output[0]["summary_text"].strip()
            print(f"[DEBUG] {filename}: Resumen generado por IA:\n{ai_summary}\n{'-'*40}")
        except Exception as e:
            print(f"[ERROR] {filename}: Error en la generación IA: {e}")
            ai_summary = narrative_summary
        
        final_summary = refine_text(ai_summary)
        print(f"[DEBUG] {filename}: Resumen final refinado:\n{final_summary}\n{'-'*40}")
        
        try:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(final_summary)
            print(f"[INFO] {filename}: Resumen guardado en: {output_path}\n{'='*50}\n")
        except Exception as e:
            print(f"[ERROR] {filename}: Error al guardar el resumen: {e}")

if __name__ == "__main__":
    process_all()
