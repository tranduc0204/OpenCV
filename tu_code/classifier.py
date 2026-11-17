"""Explainable rules; thresholds are initial values, not learned parameters."""

CLASS_NAMES = ("bolt", "washer", "nut", "unknown")
DISPLAY_NAMES = {"bolt": "Bu-lông", "washer": "Vòng đệm", "nut": "Đai ốc",
                 "unknown": "Chưa xác định"}


def classify(features, config):
    if features["elongation"] >= config.bolt_min_elongation:
        return "bolt", "Elongation exceeds bolt threshold"
    if features["elongation"] > config.round_max_elongation:
        return "unknown", "Shape lies between elongated and round rules"
    if features["hole_count"] != 1:
        return "unknown", "Expected exactly one central background-like hole"
    if (features["hex_score"] >= config.nut_min_hex_score
            and features["radial_cv"] <= config.nut_max_radial_cv):
        return "nut", "Central hole and sixfold variation of outer radius"
    if (features["hex_score"] < config.nut_min_hex_score
            and features["radial_cv"] <= config.washer_max_radial_cv):
        return "washer", "Central hole and nearly circular outer boundary"
    return "unknown", "Outer shape does not satisfy washer or nut rules"
