import pdfplumber
from transformers import pipeline


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and concatenates text from all pages of the PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_entities(text: str):
    """Uses a Hugging Face NER pipeline to extract entities from the text."""
    # Using a Spanish NER model; you can choose any appropriate model from Hugging Face
    ner_pipeline = pipeline(
        "ner",
        model="mrm8488/bert-spanish-cased-finetuned-ner",
        aggregation_strategy="simple"
    )
    entities = ner_pipeline(text)
    return entities


def structure_information(entities) -> dict:
    """
    A simple heuristic to structure extracted entities.
    In practice, you might want to fine-tune this function
    or use more advanced techniques (or even models like LayoutLM)
    to accurately map entities to your target fields.
    """
    info = {
        "metadata": {},
        "dates": [],
        "organizations": [],
        "misc": []
    }

    for entity in entities:
        label = entity.get("entity_group")
        word = entity.get("word")
        if label == "ORG":
            info["organizations"].append(word)
        elif label == "DATE":
            info["dates"].append(word)
        else:
            info["misc"].append({"label": label, "value": word})

    # Example: Assume the first organization is the issuing authority
    if info["organizations"]:
        info["metadata"]["issuing_authority"] = info["organizations"][0]

    # You can add more custom logic here to extract the title,
    # legal references, and other fields based on patterns or positions in the text.
    return info


if __name__ == "__main__":
    pdfs = [
        # 'corpus/ayudas_20-21.pdf',  # This was a scan of the original document, therefore pdfplumber fails
        'corpus/ayudas_21-22.pdf',
        'corpus/ayudas_22-23.pdf',
        'corpus/ayudas_23-24.pdf',
        'corpus/ayudas_24-25.pdf',
    ]

    for pdf in pdfs:
        print(f"Extracting text from {pdf}")

        # Extract text from PDF
        raw_text = extract_text_from_pdf(pdf)

        # Write the raw text to a file
        with open(f"corpus/{pdf.split('/')[-1].replace('.pdf', '.txt')}", "w", encoding="utf-8") as f:
            f.write(raw_text)
