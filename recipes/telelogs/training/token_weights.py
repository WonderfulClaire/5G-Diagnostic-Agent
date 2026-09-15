"""Response-only weighting for the diagnostic decision, not tool observations."""

import math
import re


def diagnosis_token_weights(text, offsets, weight=1.0):
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("diagnosis weight must be finite and at least one")
    match = re.search(r'"root_causes"\s*:\s*(\[[^\]]*\])', text)
    if match is None:
        return [1.0] * len(offsets)
    start, end = match.span(1)
    return [weight if b > a and a < end and b > start else 1.0 for a, b in offsets]
