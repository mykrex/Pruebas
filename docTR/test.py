import unicodedata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- Funciones de filtrado ---
def es_devanagari(texto):
    for char in texto:
        try:
            if 'DEVANAGARI' in unicodedata.name(char):
                return True
        except ValueError:
            pass
    return False

def es_texto_ingles(texto):
    if es_devanagari(texto):
        return False
    try:
        texto.encode('ascii')
    except UnicodeEncodeError:
        return False
    return len(texto.strip()) >= 2

# --- OCR ---
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
img_path = "public/doc3.png"
img_doc = DocumentFile.from_images(img_path)
result = model(img_doc)

# --- Visualización con bounding boxes solo en inglés ---
imagen = Image.open(img_path)
img_w, img_h = imagen.size

fig, ax = plt.subplots(1, figsize=(14, 10))
ax.imshow(imagen)

for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                if not es_texto_ingles(word.value):
                    continue  # saltar palabras no inglesas

                # geometry viene en coordenadas relativas (0 a 1)
                # formato: ((xmin, ymin), (xmax, ymax))
                (xmin, ymin), (xmax, ymax) = word.geometry

                # Convertir a píxeles
                x = xmin * img_w
                y = ymin * img_h
                w = (xmax - xmin) * img_w
                h = (ymax - ymin) * img_h

                # Dibujar bounding box
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1.5,
                    edgecolor='lime',
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Mostrar el texto encima del box
                ax.text(
                    x, y - 4,
                    word.value,
                    color='lime',
                    fontsize=7,
                    fontweight='bold',
                    backgroundcolor='black'
                )

ax.axis('off')
plt.title("Texto en inglés detectado", fontsize=13)
plt.tight_layout()
plt.show()