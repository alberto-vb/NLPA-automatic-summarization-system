import pdfplumber
import json
from transformers import pipeline

# --- Función para extraer texto del PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Pipeline de Question Answering (QA) para extracción de datos ---
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

def obtener_respuesta(pregunta, contexto):
    try:
        respuesta = qa_pipeline(question=pregunta, context=contexto)
        return respuesta['answer']
    except Exception as e:
        print(f"Error al obtener respuesta para '{pregunta}': {e}")
        return None

# --- Pipeline de Summarization para generar resumen en español ---
# Reemplaza 'model' por un modelo que funcione bien en español, por ejemplo: 'csebuetnlp/mT5_multilingual_XLSum'
summarizer = pipeline("summarization")  # Puedes especificar el modelo con el parámetro model="..."

def extraer_datos_relevantes(texto):
    datos = {}
    
    # Diccionario de preguntas para extraer información básica
    preguntas = {
        "programas": "¿Cuáles son los programas educativos a los que aplican las becas?",
        "importes": "¿Cuáles son los importes de las becas en función del rendimiento académico?",
        "umbrales": "¿Cuáles son los umbrales de ingresos establecidos?",
        "fecha_limite": "¿Cuál es la fecha límite de solicitud?"
    }
    
    # Preguntas adicionales para extraer más información
    preguntas_adicionales = {
        "requisitos": "¿Cuáles son los requisitos de acceso para las becas?",
        "modalidades": "¿Cuáles son las modalidades de las becas?",
        "beneficios": "¿Qué beneficios adicionales se mencionan en las becas?",
        "documentacion": "¿Qué documentación se requiere para la solicitud?"
    }
    
    # Combinar ambos diccionarios de preguntas
    todas_preguntas = {**preguntas, **preguntas_adicionales}
    
    # Extraer la respuesta para cada pregunta usando el pipeline QA
    for campo, pregunta in todas_preguntas.items():
        respuesta = obtener_respuesta(pregunta, texto)
        datos[campo] = respuesta
    
    # Generar un resumen del texto completo extraído
    try:
        # Si el texto es muy largo, se puede recortar o dividir para evitar errores de longitud
        texto_resumen = texto if len(texto) < 1000 else texto[:1000]
        resumen = summarizer(texto_resumen, max_length=500, min_length=150, do_sample=False)
        datos["resumen"] = resumen[0]['summary_text']
    except Exception as e:
        print(f"Error al generar resumen: {e}")
        datos["resumen"] = None

    return datos

# --- Función para guardar los datos extraídos en un archivo JSON ---
def guardar_datos_en_json(datos, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)

# --- Función principal que procesa una lista de PDFs ---
if __name__ == "__main__":
    pdfs = [
        'corpus/ayudas_21-22.pdf',
        'corpus/ayudas_22-23.pdf',
        'corpus/ayudas_23-24.pdf',
        'corpus/ayudas_24-25.pdf',
        'corpus/ayudas_25-26.pdf',
    ]

    resumen_datos = {}

    for pdf in pdfs:
        print(f"Extrayendo información de {pdf}")
        # Extraer texto del PDF
        texto = extract_text_from_pdf(pdf)

        # Extraer información relevante y resumen usando el modelo QA y el summarizer
        datos = extraer_datos_relevantes(texto)

        # Por ejemplo, se puede usar el nombre del archivo (sin extensión) como clave
        clave = pdf.split('/')[-1].replace('.pdf', '')
        resumen_datos[clave] = datos

        # Guardar la información extraída en un archivo JSON individual (opcional)
        output_json = f"corpus/{clave}.json"
        guardar_datos_en_json(datos, output_json)
        print(f"Datos guardados en {output_json}")

    # También se puede guardar todo en un único JSON
    guardar_datos_en_json(resumen_datos, "corpus/resumen_completo.json")
    print("Resumen completo guardado en corpus/resumen_completo.json")
