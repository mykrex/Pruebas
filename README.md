# Pruebas

Este primer commit es para probar y investigar asi como aprender sobre el docTR

La documentacion recomienda hacer : pip install python-doctr

´´´
Se debe trabajar con img ya que el preprocesador de Andres va a ajustar las imagenes
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
