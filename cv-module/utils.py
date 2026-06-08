# utils.py

import config

def calculate_engagement(total, angkat, hadap, menunduk=0):
    if total == 0:
        return 0.0
    
    skor_aktif   = (angkat * config.WEIGHT_TANGAN + hadap * config.WEIGHT_HADAP) / total
    penalti      = (menunduk / total) * 0.3  # menunduk kurangi skor maksimal 0.3
    score        = max(0.0, skor_aktif - penalti)
    
    # --- BAGIAN YANG DIGANTI: Mengubah skala desimal (0-1) ke persen (0-100) ---
    score_percentage = score * 100
    return round(min(score_percentage, 100.0), 2)


def smooth_engagement(prev, current):
    """Exponential moving average untuk halus-kan nilai engagement."""
    alpha = config.ENGAGEMENT_SMOOTHING
    return round(alpha * prev + (1 - alpha) * current, 2)