"""The paper's evaluation metrics, re-exported from the project-wide implementations.

ASR  - successful injection trials / total trials        (metrics.compute_trials -> "success")
SI   - Sentiment Index, 1-5                                (metrics.sentiment_index)
ΔSI  - SI_injected - SI_baseline                           (metrics.compute_trials -> "delta")
DAR  - trials with an explicit flag / total trials         (metrics.compute_trials -> "alert")
ME   - 1 - |SI_def - SI_base| / |SI_target - SI_base|       (metrics.compute_trials -> "me")
AD   - achievement density                                 (dataset.achievement_density)
"""
from dataset import achievement_density, classify_ad  # noqa: F401
from metrics import compute_trials, sentiment_index, summarize  # noqa: F401


def bias_magnitude(si_injected: float, si_baseline: float) -> float:
    return si_injected - si_baseline


def mitigation_efficiency(si_defended: float, si_baseline: float, si_target: float) -> float | None:
    denom = abs(si_target - si_baseline)
    if denom == 0:
        return None
    return max(0.0, min(1.0, 1 - abs(si_defended - si_baseline) / denom))
