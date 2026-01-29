from pathlib import Path
print("RUNNING:", Path(__file__).resolve())
from .config import DATA_DIR

import torch
from .NLLsurv import NLLSurvLoss
from .metric import manual_concordance_index, cumulative_risk_matrix



@torch.no_grad()
def evaluate_model(model, X_test, Y_test, tau, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # 1) Predict
    probs = model(X_test)  # (N, T)

    # 2) Test loss
    loss_fn = NLLSurvLoss()
    test_loss = loss_fn(probs, Y_test).item()

    # 3) Compute Fhat(tau)
    risk_mat = cumulative_risk_matrix(probs)                 # (N, T)


    # 4) C-index at tau
    y_true = Y_test.detach().cpu().numpy()   # (N, 2)
    c_td = manual_concordance_index(y_true, risk_mat.detach().cpu().numpy())

    return test_loss, c_td
