"""
Entry point — run from the project root:
  python main.py
"""

import json
from pathlib import Path

from ocr import Config, PaddleOCRAdapter, KIEEngine, DolphinOCRAdapter, DotsOCRAdapter
from ocr.pipeline import load_mapping, process


def main() -> None:
    cfg     = Config()
    mapping = load_mapping(config=cfg)
    #ocr     = PaddleOCRAdapter(cfg)
    #ocr = DolphinOCRAdapter(cfg, model_path="./ocr/hf_model", dolphin_repo="./ocr/Dolphin")
    ocr = DotsOCRAdapter(cfg, model_path="./ocr/DotsOCR")
    kie     = KIEEngine(mapping)

    # Default sample image — adjust path or accept as CLI arg as needed.
    image_path = Path(__file__).parent / "ocr" / "public" / "originals" / "id.jpg"
    result = process(image_path, ocr, kie, cfg)

    print("\n── FINAL RESULT ──")
    print(f"Document type: {result['document_type']}")
    print(json.dumps(result["fields"], indent=2, ensure_ascii=False))

    print("\n── ALL DETECTED LINES ──")
    for line in result["raw_lines"]:
        print(f"[{line.lang:8}] [{line.confidence:.2f}] {line.text}")


if __name__ == "__main__":
    main()
