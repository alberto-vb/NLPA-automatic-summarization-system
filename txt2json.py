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

def spacy_extractor(text, max_entities=10, max_sentence_length=120):
    nlp = spacy.load("es_core_news_md")
    doc = nlp(text)

    entidades = {
        "FECHAS": [],
        "DINERO": [],
        "ORG": []
    }

    # Función para acortar frases manteniendo el contexto
    def shorten_sentence(sentence, max_length):
        if len(sentence) <= max_length:
            return sentence
        words = sentence.split()
        # Mantener la entidad y palabras alrededor
        for ent in doc.ents:
            if ent.text in sentence:
                start = max(0, sentence.find(ent.text) - 20)
                end = min(len(sentence), sentence.find(ent.text) + len(ent.text) + 20)
                return sentence[start:end] + "..."
        return ' '.join(words[:15]) + '...' if len(words) > 15 else sentence

    # Contadores para limitar entidades
    counters = {"FECHAS": 0, "DINERO": 0, "ORG": 0}

    # Extraer entidades con frases relevantes
    for sent in doc.sents:
        for ent in sent.ents:
            ent_type = None
            
            if ent.label_ == "DATE" and counters["FECHAS"] < max_entities:
                ent_type = "FECHAS"
            elif (ent.label_ == "MONEY" or 
                  any(t.like_num and "€" in sent.text for t in sent) or 
                  any(t.like_num and "euro" in sent.text.lower() for t in sent)) and counters["DINERO"] < max_entities:
                ent_type = "DINERO"
            elif ent.label_ == "ORG" and counters["ORG"] < max_entities:
                ent_type = "ORG"
                
            if ent_type:
                entidades[ent_type].append({
                    "entidad": ent.text,
                    "frase": shorten_sentence(sent.text, max_sentence_length),
                    "tipo": ent.label_
                })
                counters[ent_type] += 1

    # Extraer frases de acción más relevantes (limitadas a 3)
    verbos_accion = ["convocar", "requerir", "solicitar", "otorgar", "financiar"]
    frases_accion = []
    
    for sent in doc.sents:
        if len(frases_accion) >= 8:  # Máximo 8 frases de acción
            break
            
        action_verbs = [token.lemma_ for token in sent if token.lemma_ in verbos_accion]
        if action_verbs:
            frases_accion.append({
                "frase": shorten_sentence(sent.text, max_sentence_length),
                "verbos": list(set(action_verbs))[:2]  # Máximo 2 verbos por frase
            })

    # Eliminar duplicados y asegurar formato compacto
    resultados = {
        "entidades": {k: v[:max_entities] for k, v in entidades.items()},  # Asegurar máximo de entidades
        "frases_accion": frases_accion
    }
    
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
        'corpus/ayudas_24-25.txt',
        'corpus/ayudas_25-26.txt'
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
                # Guardar resultados de spacy_extractor
                save_json(spacy_extractor(contenido), "resumenes_spacy", txt)

        except FileNotFoundError:
            print(f"Error: El archivo {txt} no fue encontrado")
        except Exception as e:
            print(f"Error al procesar el archivo {txt}: {str(e)}")