"""MTCNN face detector placeholder.

This module contains minimal placeholder functions used by the FaceNet
scaffold. Implement detection logic (e.g., using MTCNN) when you're ready.
"""

from typing import List, Tuple


def detect_faces(image) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the provided image.

    Args:
        image: Image array (e.g., numpy ndarray or PIL Image).

    Returns:
        A list of bounding boxes as (x, y, width, height).

    NOTE: This is a placeholder. Replace with actual detector implementation.
    """
    raise NotImplementedError("detect_faces() is a placeholder — implement using MTCNN or another detector.")
