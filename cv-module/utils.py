# utils.py

import config

def calculate_engagement(total, angkat, hadap):
    """
    Hitung engagement score (0.0 - 1.0).
    Gunakan smoothing di main.py agar tidak jumping.
    """
    if total == 0:
        return 0.0
    score = (angkat * config.WEIGHT_TANGAN + hadap * config.WEIGHT_HADAP) / total
    return round(min(score, 1.0), 2)


def smooth_engagement(prev, current):
    """Exponential moving average untuk halus-kan nilai engagement."""
    alpha = config.ENGAGEMENT_SMOOTHING
    return round(alpha * prev + (1 - alpha) * current, 2)