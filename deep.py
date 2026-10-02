"""Tabular neural network (PyTorch) wrapped as a scikit-learn classifier, so it works inside Pipeline,
cross_val_predict, etc.  Requires: pip install torch   (optional - train.py skips it if torch is missing).
Design: MLP + BatchNorm + Dropout, AdamW (weight decay), early stopping on a held-out validation split."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=60,
                 batch_size=128, patience=8, val_fraction=0.15, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        perm = np.random.default_rng(self.random_state).permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        va, tr = perm[:n_val], perm[n_val:]
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        net = nn.Sequential(*layers)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])
        best, best_state, bad, epoch = float("inf"), None, 0, 0
        for epoch in range(self.epochs):
            net.train()
            order = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                b = order[i:i + self.batch_size]
                if len(b) < 2:          # BatchNorm needs more than one row
                    continue
                opt.zero_grad()
                loss_fn(net(Xt[b]).squeeze(1), yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xv).squeeze(1), yv).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_, self.classes_, self.n_epochs_ = net, np.array([0, 1]), epoch + 1
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
