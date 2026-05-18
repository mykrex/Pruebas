from pathlib import Path

import numpy as np
from PIL import Image as PILImage


def load_image(path: str | Path, max_side: int = 1_400) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        print(f"  Resized: {w}x{h} → {new_w}x{new_h}")
    return np.array(img)
