"""Reference-plane maps passed to planning and to the synchronized viewer."""

import base64
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = (68, 54)
MAX_VALUE = 250
OUTSIDE, UNKNOWN = 254, 255
VIRTUAL_GREEN, VIRTUAL_CONTRAST = 103., 64.


def grid_from_plane(radii, size=SIZE):
    a, b = radii
    w, h = size
    return np.array([[w / (2 * a), 0, (w - 1) / 2],
                     [0, -h / (2 * b), (h - 1) / 2], [0, 0, 1.]])


@dataclass
class SurfaceField:
    values: np.ndarray
    threshold: float
    kind: str

    @property
    def coverage(self):
        inside = self.values < OUTSIDE
        return float(np.count_nonzero(inside & (self.values >= self.threshold * MAX_VALUE)) / max(1, inside.sum()))

    @property
    def mean(self):
        valid = self.values < OUTSIDE
        return float(self.values[valid].mean() / MAX_VALUE) if np.any(valid) else None

    def view(self):
        h, w = self.values.shape
        return {"width": w, "height": h, "data": base64.b64encode(self.values.tobytes()).decode("ascii"),
                "coverage": self.coverage, "mean": self.mean, "kind": self.kind,
                "known_fraction": float(np.count_nonzero(self.values < OUTSIDE) / max(1, np.count_nonzero(self.values != OUTSIDE)))}


def project_field(pixels, image_to_plane, radii, threshold, kind):
    if image_to_plane is None:
        return None
    mapping = grid_from_plane(radii) @ image_to_plane
    field = cv2.warpPerspective(pixels, mapping, SIZE, flags=cv2.INTER_NEAREST, borderValue=UNKNOWN)
    h, w = field.shape
    y, x = np.ogrid[:h, :w]
    outside = ((x + .5) * 2 / w - 1) ** 2 + ((y + .5) * 2 / h - 1) ** 2 > 1
    field[outside] = OUTSIDE
    return SurfaceField(field, threshold, kind)
