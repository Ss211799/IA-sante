from pathlib import Path
import torch

print("RUNNING:", Path(__file__).resolve())


import numpy as np
import torch


def cumulative_risk_matrix(probs: torch.Tensor) -> torch.Tensor:
    """
    Matrice de risque cumulé à partir des probabilités discrètes.

    probs: (N, T) avec probs[i, t] = P(T_i = t+1) (si temps discrets 1..T)
    return: (N, T) avec risk[i, t] = P(T_i <= t+1)
    """
    return probs.cumsum(dim=1)


def manual_concordance_index(y_true, risk_matrix):
    """
    Time-dependent C-index (version "ancienne" à 2 arguments).

    Args:
        y_true: array (N, 2)
            y_true[:, 0] = time to event/censor (discret)
            y_true[:, 1] = event indicator (1=event, 0=censored)
        risk_matrix: array (N, T)
            cumulative risk scores: risk[i, t] = P(T_i <= t)

    Returns:
        c_index: float in [0, 1]
    """
    y_true = np.asarray(y_true)
    risk_matrix = np.asarray(risk_matrix)

    times = y_true[:, 0].astype(int)
    events = y_true[:, 1].astype(int)

    concordant_pairs = 0.0
    total_comparable_pairs = 0.0
    n = len(times)

    for i in range(n):
        if events[i] == 1:  # patient i décédé
            t_i = int(times[i])

			# On utilise le risque cumulé au temps t_i
            t_idx = min(max(t_i - 1, 0), risk_matrix.shape[1] - 1)

            for j in range(n):
                if times[j] > times[i]:  # j survit plus longtemps que i
                    total_comparable_pairs += 1.0

                    # Comparer les risques au temps t_i
                    risk_i = risk_matrix[i, t_idx]
                    risk_j = risk_matrix[j, t_idx]

                    if risk_i > risk_j:
                        concordant_pairs += 1.0
                    elif risk_i == risk_j:
                        concordant_pairs += 0.5

    return (concordant_pairs / total_comparable_pairs) if total_comparable_pairs > 0 else 0.0
