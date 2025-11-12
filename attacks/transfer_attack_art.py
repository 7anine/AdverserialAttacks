from pathlib import Path
import numpy as np
import joblib
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent

# Load test data (force float32 for PyTorch)
X_test = np.load(BASE / "X_test_scaled.npy").astype(np.float32)
y_test = np.load(BASE / "y_test_binary.npy")

# Load trained RandomForest (target model)
rf = joblib.load(BASE / "rf_model.joblib")
rf_art = SklearnClassifier(model=rf)

# Recreate surrogate MLP and load weights
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_test.shape[1]
surrogate = MLP(input_dim).to(device)
surrogate.load_state_dict(torch.load(BASE / "surrogate_mlp.pth", map_location=device))
surrogate.eval()

# Wrap the surrogate with ART
loss_fn = nn.CrossEntropyLoss()
x_min = float(X_test.min())
x_max = float(X_test.max())

surrogate_art = PyTorchClassifier(
    model=surrogate,
    loss=loss_fn,
    input_shape=(input_dim,),
    nb_classes=2,
    clip_values=(x_min, x_max),
)

# Baseline accuracy on clean test set
preds_clean_rf = rf.predict(X_test)
print(f"RF clean acc: {accuracy_score(y_test, preds_clean_rf):.4f}")

# Attack configurations (increase eps for stronger attacks)
attacks = {
    "FGSM_eps_0.075": FastGradientMethod(estimator=surrogate_art, eps=0.075),
    "FGSM_eps_0.125": FastGradientMethod(estimator=surrogate_art, eps=0.125),
    "PGD_eps_0.075": ProjectedGradientDescent(estimator=surrogate_art, eps=0.075, max_iter=20, eps_step=0.005),
    "PGD_eps_0.125": ProjectedGradientDescent(estimator=surrogate_art, eps=0.125, max_iter=40, eps_step=0.01),
}

batch_size = 2048
n = X_test.shape[0]
results = {}

for name, attack in attacks.items():
    print(f"\nRunning attack {name} ...")
    X_adv = np.zeros_like(X_test, dtype=np.float32)

    # Generate adversarial examples in batches
    for i in range(0, n, batch_size):
        xb = X_test[i:i+batch_size].astype(np.float32)
        xb_adv = attack.generate(x=xb).astype(np.float32)
        X_adv[i:i+batch_size] = xb_adv

    # Save adversarial set
    np.save(BASE / f"X_test_adv_{name}.npy", X_adv)

    # Evaluate the transferred adversarial examples on the RandomForest
    preds_adv_rf = rf.predict(X_adv)
    acc_adv_rf = accuracy_score(y_test, preds_adv_rf)

    # Adversarial Success Rate (ASR): fraction of malicious samples that evaded detection
    malicious_mask = (y_test == 1)
    if malicious_mask.sum() > 0:
        before_detected = (preds_clean_rf[malicious_mask] == 1).sum()
        after_detected = (preds_adv_rf[malicious_mask] == 1).sum()
        evaded = before_detected - after_detected
        asr = evaded / float(malicious_mask.sum())
    else:
        asr = None

    print(f"{name} --> RF acc on adv: {acc_adv_rf:.4f}  ASR: {asr}")
    results[name] = {"acc_adv_rf": acc_adv_rf, "ASR": asr}

# Print final summary
print("\nSummary:")
for k, v in results.items():
    print(k, v)
