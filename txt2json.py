# pip install nltk
#pip install spacy
#python -m spacy download es_core_news_md


import re
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Descargar recursos de NLTK (solo primera vez)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Download the Spanish tokenizer data

def date_and_money_extractor(text, filename):
    # EXTRACTOR DE LAS FRASES SIN REPETIDAS, SIN STOPWORDS, QUE CONTENGAN FECHAS Y DINERO

    # 1. Extraer frases con fechas y montos
    frases_con_fechas = re.findall(
        r'([^.]*?\d{1,2}\sde\s\w+\sde\s\d{4}[^.]*\.|[^.]*?\d{2}/\d{2}/\d{4}[^.]*\.)',
        text,
        re.IGNORECASE
    )

    frases_con_montos = re.findall(
        r'([^.]*?\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:euros|€)[^.]*\.)',
        text,
        re.IGNORECASE
    )

    # 2. Función para quitar stopwords (conservando números y símbolos)
    def quitar_stopwords(frase):
        stop_words = set(stopwords.words('spanish'))
        palabras = word_tokenize(frase.lower(), language='spanish')
        palabras_filtradas = []
        for palabra in palabras:
            if (palabra not in stop_words) and (palabra.isalpha() or re.match(r'[\d€€/.,-]', palabra)):
                palabras_filtradas.append(palabra)
        return ' '.join(palabras_filtradas)

    # 3. Aplicar limpieza y eliminar duplicados
    frases_fechas_limpias = list(set([quitar_stopwords(frase) for frase in frases_con_fechas]))
    frases_montos_limpias = list(set([quitar_stopwords(frase) for frase in frases_con_montos]))

    # 4. Resultados
    resultados = {
        "fechas": frases_fechas_limpias,
        "montos": frases_montos_limpias
    }

    return resultados

def date_semantic_pattern(text, filename):
    # Extraer secciones clave usando patrones semánticos
    patrones = {
        "convocatoria": r"CONVOCATORIA.*?(?=\n|$)",
        "requisitos": r"Requisitos:(.*?)(?=\n\w+:|$)",
        "plazo": r"Plazo: (\d{2}/\d{2}/\d{4}-\d{2}/\d{2}/\d{4})"
    }

    resultados = {}
    for nombre, patron in patrones.items():
        matches = re.findall(patron, text, re.DOTALL | re.IGNORECASE)
        resultados[nombre] = [match.strip() for match in matches]

    return resultados

def save_json(data, foldername, filename):
    # Crear carpeta resumenes si no existe
    os.makedirs(foldername, exist_ok=True)

    # Generar nombre del archivo de salida
    nombre_base = os.path.splitext(os.path.basename(filename))[0]
    output_file = f"{foldername}/{nombre_base}_resumen.json"

    # Guardar resultados en JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Resumen guardado en: {output_file}")

if __name__ == "__main__":
    txts = [
        'corpus/ayudas_21-22.txt',
        'corpus/ayudas_22-23.txt',
        'corpus/ayudas_23-24.txt',
        'corpus/ayudas_24-25.txt',
        'corpus/ayudas_25-26.txt',
    ]

    for txt in txts:
        print("_"*40)
        print(f"Procesando archivo: {txt}")
        print("_"*40)
        
        try:
            with open(txt, 'r', encoding='utf-8') as file:
                contenido = file.read()
                # Guardar resultados de date_and_money_extractor
                save_json(date_and_money_extractor(contenido, txt), "resumenes_regex", txt)
                # Guardar resultados de date_semantic_pattern
                save_json(date_semantic_pattern(contenido, txt), "resumenes_semantic", txt)

        except FileNotFoundError:
            print(f"Error: El archivo {txt} no fue encontrado")
        except Exception as e:
            print(f"Error al procesar el archivo {txt}: {str(e)}")