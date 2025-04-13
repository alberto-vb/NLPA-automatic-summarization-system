import os

print("\n=== BOE Scholarship Summarizer ===")

# 1. EXTRAER TEXTO DE LOS PDFs
print("\n[1/5] Extrayendo texto de los PDFs...")
os.system("python extract_text.py")

# 2. LOCALIZAR SECCIONES IMPORTANTES
print("\n[2/5] Localizando secciones clave...")
os.system("python locate_sections.py")

# 3. PARSEAR INFORMACIÓN CLAVE
print("\n[3/5] Parseando información...")
os.system("python parse_sections.py")

# 4. GENERAR RESÚMENES INDIVIDUALES CON BERT
print("\n[4/5] Generando resúmenes individuales...")
os.system("python generate_summary_exp1.py")
os.system("python generate_summary_exp2.py")
os.system("python generate_summary_exp3.py")
os.system("python generate_summary_exp4.py")

print("\n=== PIPELINE COMPLETO ===")
print("Todos los outputs están en:")
print(" - data/extracted/")
print(" - data/sections/")
print(" - data/parsed/")
print(" - data/summaries/")

print("\n¡Hecho!")
