def calculate_engagement(total, angkat, hadap):
    if total == 0:
        return 0.0

    score = (angkat * 0.6 + hadap * 0.4) / total
    return round(score, 2)