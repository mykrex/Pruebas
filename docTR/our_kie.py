from doctr.io import DocumentFile
from doctr.models import kie_predictor
import matplotlib.pyplot as plt 

# Model for Key Information Extraction
model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
# Pasamos la img
doc = DocumentFile.from_images("public/doc3.png")
# Analisis del doc con el modelo KIE
result = model(doc)

# Display the result of the model
predictions = result.pages[0].predictions
for class_name in predictions.keys():
    list_predictions = predictions[class_name]
    for prediction in list_predictions:
        print(f"Prediction for {class_name}: {prediction}")

# Sintetizar lo que se ha detectado y reconocido, y mostrarlo
synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()

## PROBAR CON spaCy para el análisis de las entidades reconocidas