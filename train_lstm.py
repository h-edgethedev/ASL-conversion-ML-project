import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEQ_LEN = 10
FEAT_DIM = 63

DATA_PATH = "models/data/seq_dataset.csv"
MODEL_PATH = "models/lstm_model.pt"

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)         # out: (B, T, H)
        last = out[:, -1, :]          # last timestep
        logits = self.fc(last)        # (B, C)
        return logits

def main():
    df = pd.read_csv(DATA_PATH)

    X_flat = df.drop(columns=["label", "person_id"], errors="ignore").values
    y_text = df["label"].values

    # reshape (N, 630) -> (N, 10, 63)
    X = X_flat.reshape(-1, SEQ_LEN, FEAT_DIM)

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = SeqDataset(X_train, y_train)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(num_classes=len(le.classes_)).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1:02d} | loss {total_loss:.3f} | test acc {acc:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": le.classes_.tolist()},
        MODEL_PATH
    )
    print("Saved model to:", MODEL_PATH)

if __name__ == "__main__":
    main()