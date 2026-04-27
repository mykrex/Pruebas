# Pruebas

Se debe trabajar con img

---

## Flujo

OCR[Extrae los caracteres] -> Separar caracteres [Latinos][Otros] -> KIE [Identificacion de entidades]

---

## OCR

### DocTR

El primero recomendado

pip install "python-doctr[torch]" <- Debido a que trabajaremos con PyTorch

pip install matplotlib mplcursors <- Solo para try.py y kie.py

### Paddle

```python
pip install "paddlepaddle>=3.0"

pip install paddleocr
```

### Dolphin

```python
pip install huggingface_hub

huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model

pip install torch torchvision accelerate qwen_vl_utils opencv-python Pillow pymupdf

pip install "transformers==4.51.0"
```

### Notas del modelo

---

## KIE

### spaCy

Para el análisis de las entidades reconocidas

```python
pip install spacy

python -m spacy download en_core_web_sm
```

---

## Otros

```python
pip install opencv-python
```

---

## Alternativa para identificar los datos del documento

Donut (Document Understanding Transformer)
No necesita OCR separado — lee la imagen directamente y extrae campos estructurados de una vez

---

## METRICAS

- Tiempo de procesamiento
- Tiempo de respuesta
- Certeza / Confiabilidad en el resultado

---

## NOTAS

- ¿Todos los modelos de OCR usaran CPU?
- Redimencionar las imagenes para que la maquina pueda trabajar con ella con tal OCR
- KIE : modelo o librerias como spaCy segun sea el caso
- ¿Un dataset sería útil aquí?
  - No para esta parte. Un dataset sería útil si quisieras fine-tunear Dolphin para que reconozca mejor scripts indios o campos específicos de los documentos. Para inferencia básica no se necesita eso. Dolphin ya viene preentrenado [yasehace].
