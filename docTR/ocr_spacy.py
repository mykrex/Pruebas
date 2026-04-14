from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import spacy

# Modelo de OCR
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Pasamos la img
doc = DocumentFile.from_images("public/doc3.png")
# Analisis del doc con el modelo KIE
result = model(doc)

# Extraer caracteres latinos

def is_latin(text):
    latin = sum(1 for c in text if '\u0000' <= c <= '\u024F')
    return latin / max(len(text), 1) > 0.7

english_words = []
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                if is_latin(word.value):
                    english_words.append(word.value)

english_text = " ".join(english_words)
print("Texto en inglés extraído:", english_text)

# Identificar identidades con spaCy

nlp = spacy.load("en_core_web_sm")
doc_nlp = nlp(english_text)

for ent in doc_nlp.ents:
    print(f"{ent.label_:<12} → {ent.text}")
