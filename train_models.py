# train_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load preprocessed data
X_train = np.load("X_train_scaled.npy")
y_train = np.load("y_train_binary.npy")
X_test = np.load("X_test_scaled.npy")
y_test = np.load("y_test_binary.npy")

# Train a RandomForest detector and save it
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("RandomForest test acc:", accuracy_score(y_test, y_pred_rf))
joblib.dump(rf, "rf_model.joblib")

# Small MLP surrogate (used for generating adversarial examples)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)   # binary output
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = MLP(input_dim).to(device)

# Create DataLoader for training
bs = 256
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        acc = (preds == y_test).mean()

    print(f"Epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f} test_acc={acc:.4f}")

# Save surrogate weights
torch.save(model.state_dict(), "surrogate_mlp.pth")
print("Saved surrogate_mlp.pth and rf_model.joblib")
