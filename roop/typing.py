# roop/typing.py

from typing import TypeAlias
import numpy as np
from insightface.app.common import Face as InsightFace

# Type alias for a face object from InsightFace
Face: TypeAlias = InsightFace

# Type alias for a video or image frame as a NumPy array
# Typically uint8 with shape (height, width, 3) for RGB images
Frame: TypeAlias = np.ndarray[tuple[int, int, 3], np.uint8]
