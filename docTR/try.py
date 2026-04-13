from doctr.io import DocumentFile # Para cargar documentos
from doctr.models import ocr_predictor #Importamos el predictor de DocTR
from doctr.models import kie_predictor # KIE predictor, puede extraer entidades de los docs

import matplotlib.pyplot as plt # Probar con Matplotlib

# Creamos el modelo con el predictor
# Para text detection usamos el modelo db_resnet50
# Para text recognition usamos el modelo crnn_vgg16_bn
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Se pude leer 1 o mas imagenes, asi como PDFs con multiples paginas
img_doc = DocumentFile.from_images("public/minecraft_wikipedia.png")

# Hacer la prediccion
result = model(img_doc)

# Display the result of the model
result.show()

# Sintetizar lo que se ha detectado y reconocido, y mostrarlo
synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()

# The ocr_predictor returns a Document object with a nested structure
# (with Page, Block, Line, Word, Artefact)
# Se puede exportar el resultado en JSON
json_output = result.export()
print(json_output)