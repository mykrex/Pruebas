# Pruebas

Este primer commit es para probar y investigar asi como aprender sobre el docTR

La documentacion recomienda hacer : pip install python-doctr

´´´
Se debe trabajar con img
´´´

# Maquina virtual para probarlo:

python -m venv venv

# Dependencias

pip install "python-doctr[torch]" <- Esto lo hice asi por recomendacion debido a que trabajaremos con PyTorch

pip install matplotlib mplcursors

# Flujo

OCR -> Filtrar caracteres latinos [Ingles] -> spaCy

# spaCy

Para el análisis de las entidades reconocidas

´´´

pip install spacy

python -m spacy download en_core_web_sm

´´´

# Alternativa para identificar los datos del documento

Donut (Document Understanding Transformer)
No necesita OCR separado — lee la imagen directamente y extrae campos estructurados de una vez.

# Prueba con PaddleOCR

pip install "paddlepaddle>=3.0"

pip install paddleocr

# OTROS

pip install opencv-python

## Pendiente:

´´´

Separar entre los caracteres latinos y no latinos, probar con los preprocesados ya

´´´

# REDIMENSIONAR

- ¿Si uso el de easy ocuparia redimensionar de igual forma?
- Con la tarjeta y la visa:
  - SACAR IDENTIDADES CON:
    - REGEX
    - SPACY
  - CON :
    - EASY
    - PADDLE

- PADDLE + REGEX
- PADDLE + SPACY
- EASY + REGEX
- EASY + SPACY
