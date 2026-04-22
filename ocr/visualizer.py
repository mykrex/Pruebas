import cv2
import numpy as np

from .schema import TextLine


_LANG_COLORS: dict[str, tuple[int, int, int]] = {
    "english": ( 34, 197,  94),  # green
    "indian":  (  0, 165, 255),  # orange
    "unknown": (160, 160, 160),  # grey
}


def visualize(
    image_rgb: np.ndarray,
    groups: dict[str, list[TextLine]],
    output_path: str,
) -> None:
    """Draw colour-coded bounding boxes and save to output_path."""
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for lang, lines in groups.items():
        color = _LANG_COLORS[lang]
        for line in lines:
            cv2.polylines(vis, [line.bbox], isClosed=True, color=color, thickness=2)
            origin = (int(line.bbox[0][0]), int(line.bbox[0][1]) - 5)
            cv2.putText(vis, line.text[:30], origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(output_path, vis)
    print(f"\n  Saved: {output_path}")
    print("  Green=English | Orange=Indian script | Grey=numbers/symbols")
