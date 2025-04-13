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

def regex_extractor(text):
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

def semantic_pattern_extractor(text):
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

def spacy_extractor(text):
    nlp = spacy.load("es_core_news_md")
    doc = nlp(text)

    entidades = {
        "FECHAS": [],
        "DINERO": [],
        "ORG": []
    }

    # Extraer frases completas que contengan entidades
    for sent in doc.sents:
        for ent in sent.ents:
            # Para fechas: mejorar detección incluyendo patrones comunes
            if ent.label_ == "DATE":
                entidades["FECHAS"].append({
                    "entidad": ent.text,
                    "frase": sent.text,
                    "tipo": "DATE"
                })
            
            # Para dinero: ampliar criterios de detección
            elif ent.label_ == "MONEY" or \
                 any(t.like_num and "€" in sent.text for t in sent) or \
                 any(t.like_num and "euro" in sent.text.lower() for t in sent):
                entidades["DINERO"].append({
                    "entidad": ent.text,
                    "frase": sent.text,
                    "tipo": "MONEY"
                })
            
            # Para organizaciones (ya funciona bien)
            elif ent.label_ == "ORG":
                entidades["ORG"].append({
                    "entidad": ent.text,
                    "frase": sent.text,
                    "tipo": "ORG"
                })

    # Mejorar extracción de frases con verbos de acción
    verbos_accion = ["convocar", "requerir", "solicitar", "otorgar", "financiar"]
    frases_accion = [
        {
            "frase": sent.text,
            "verbos": [token.lemma_ for token in sent if token.lemma_ in verbos_accion]
        }
        for sent in doc.sents
        if any(token.lemma_ in verbos_accion for token in sent)
    ]

    # Filtrar duplicados manteniendo el orden
    for key in entidades:
        seen = set()
        entidades[key] = [x for x in entidades[key] if not (x["frase"] in seen or seen.add(x["frase"]))]

    resultados = {
        "entidades": entidades,
        "frases_accion": frases_accion
    }
    
    #print(json.dumps(resultados, indent=2, ensure_ascii=False))
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
        #'corpus/ayudas_24-25.txt',
    ]

    for txt in txts:
        print("_"*40)
        print(f"Procesando archivo: {txt}")
        print("_"*40)
        
        try:
            with open(txt, 'r', encoding='utf-8') as file:
                contenido = file.read()
                # Guardar resultados de regex_extractor
                #save_json(regex_extractor(contenido), "resumenes_regex", txt)
                # Guardar resultados de semantic_pattern_extractor
                #save_json(semantic_pattern_extractor(contenido), "resumenes_semantic", txt)
                save_json(spacy_extractor(contenido), "resumenes_spacy", txt)

        except FileNotFoundError:
            print(f"Error: El archivo {txt} no fue encontrado")
        except Exception as e:
            print(f"Error al procesar el archivo {txt}: {str(e)}")