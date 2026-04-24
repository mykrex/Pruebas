import json
from pathlib import Path

from .config import Config
from .engine import OCREngine
from .image import load_image
from .kie import KIEEngine
from .language import classify_lines
from .visualizer import visualize


def load_mapping(path: str | Path | None = None, config: Config | None = None) -> dict:
    resolved = Path(path) if path else (config or Config()).fields_json
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


def process(
    image_path: str | Path,
    ocr: OCREngine,
    kie: KIEEngine,
    config: Config,
    save_viz: bool = True,
) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\n{'='*50}\nProcessing: {path.name}\n{'='*50}")

    print("→ Stage 1: Loading & OCR...")
    image = load_image(path, config.max_image_side)
    lines = ocr.extract(image)
    print(f"  {len(lines)} lines (confidence >= {config.confidence_threshold})")

    print("→ Stage 2: Language classification...")
    groups = classify_lines(lines)
    print(
        f"  English: {len(groups['english'])} | "
        f"Indian: {len(groups['indian'])} | "
        f"Unknown: {len(groups['unknown'])}"
    )

    print("→ Stage 3: KIE...")
    result = kie.extract(groups["english"], groups["unknown"])
    print(f"  Document type : {result['document_type']}")
    print(f"  Fields found  : {list(result['fields'].keys())}")

    if save_viz:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        out = str(config.output_dir / f"{path.stem}_result.jpg")
        print("→ Saving visualisation...")
        visualize(image, groups, out)

    return {
        "path":             str(path),
        "raw_lines":        lines,
        "language_summary": {k: len(v) for k, v in groups.items()},
        "document_type":    result["document_type"],
        "fields":           result["fields"],
    }
