# logic_calibrator.py
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class BeliefDataset(Dataset):
    """
    Dataset trên raw belief vector v ∈ R^5 và label y ∈ {0,1}.
    v = [p_c, p_para, p_weak, p_str, p_neg].
    """

    def __init__(self, V: np.ndarray, y: np.ndarray):
        assert V.shape[1] == 5, f"Expect belief dim=5, got {V.shape[1]}"
        self.V = torch.from_numpy(V.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.V.shape[0]

    def __getitem__(self, idx):
        return self.V[idx], self.y[idx]


class LogicNet(nn.Module):
    """
    MLP nhận v (5-D) -> logits_hat (5-D).
    Probabilities \hat v = sigmoid(logits_hat) ∈ (0,1)^5.
    """

    def __init__(self, belief_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (batch, 5)
        return logits_hat: (batch, 5)
        """
        return self.net(v)


class LogicConstrainedCalibrator:
    """
    E1: Logic-constrained calibrator trên raw belief vector.

    Input:
        V: (N, 5) belief vectors từ LLM
        y: (N,) labels

    Model:
        v -> logits_hat -> \hat v = sigmoid(logits_hat)

    Loss:
        - BCE(\hat p_c, y)
        - Logic:
            L_neg  = | \hat p_c + \hat p_neg - 1 |
            L_weak = max(0, \hat p_c - \hat p_weak)
            L_str  = max(0, \hat p_str - \hat p_c)
        - Optional regularizer:
            L_reg = || \hat v - v ||^2

        Total:
            L = w_bce * L_bce + w_logic * L_logic + w_reg * L_reg
    """

    def __init__(
        self,
        alpha_neg: float = 1.0,
        alpha_weak: float = 0.5,
        alpha_str: float = 0.5,
        w_bce: float = 1.0,
        w_logic: float = 1.0,
        w_reg: float = 0.1,
        belief_dim: int = 5,
        hidden_dim: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.alpha_neg = alpha_neg
        self.alpha_weak = alpha_weak
        self.alpha_str = alpha_str
        self.w_bce = w_bce
        self.w_logic = w_logic
        self.w_reg = w_reg
        self.device = device

        self.model = LogicNet(belief_dim=belief_dim, hidden_dim=hidden_dim).to(self.device)
        self._fitted = False

    def _logic_loss(self, v_orig: torch.Tensor, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        v_orig, v_hat: (batch, 5), both in [0,1].
        Return:
            total_logic_loss, loss_neg, loss_weak, loss_str
        """
        p_c = v_hat[:, 0]
        p_para = v_hat[:, 1]
        p_weak = v_hat[:, 2]
        p_str = v_hat[:, 3]
        p_neg = v_hat[:, 4]

        # Logic constraints
        viol_neg = torch.abs(p_c + p_neg - 1.0)
        viol_weak = torch.clamp(p_c - p_weak, min=0.0)
        viol_str = torch.clamp(p_str - p_c, min=0.0)

        loss_neg = viol_neg.mean()
        loss_weak = viol_weak.mean()
        loss_str = viol_str.mean()

        logic_loss = (
            self.alpha_neg * loss_neg
            + self.alpha_weak * loss_weak
            + self.alpha_str * loss_str
        )

        return logic_loss, loss_neg, loss_weak, loss_str

    def fit(
        self,
        V: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 20,
        verbose: bool = True,
    ):
        """
        V: (N, 5) raw beliefs from LLM.
        y: (N,) labels in {0,1}.
        """
        dataset = BeliefDataset(V, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_bce = 0.0
            total_logic = 0.0
            total_reg = 0.0
            n_batches = 0

            for v_batch, y_batch in loader:
                v_batch = v_batch.to(self.device)          # (B,5), original beliefs
                y_batch = y_batch.to(self.device)          # (B,)

                optimizer.zero_grad()

                logits_hat = self.model(v_batch)           # (B,5)
                v_hat = torch.sigmoid(logits_hat)          # (B,5)

                # Supervision on \hat p_c
                logits_c = logits_hat[:, 0]
                bce = bce_loss_fn(logits_c, y_batch)

                # Logic loss on v_hat
                logic_loss, loss_neg, loss_weak, loss_str = self._logic_loss(v_batch, v_hat)

                # Regularizer: keep v_hat close to v_batch
                reg_loss = torch.mean((v_hat - v_batch) ** 2)

                loss = (
                    self.w_bce * bce
                    + self.w_logic * logic_loss
                    + self.w_reg * reg_loss
                )

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_bce += float(bce.item())
                total_logic += float(logic_loss.item())
                total_reg += float(reg_loss.item())
                n_batches += 1

            if verbose:
                print(
                    f"[LogicCalib] Epoch {epoch+1}/{epochs} "
                    f"loss={total_loss/n_batches:.4f} "
                    f"bce={total_bce/n_batches:.4f} "
                    f"logic={total_logic/n_batches:.4f} "
                    f"reg={total_reg/n_batches:.4f}"
                )

        self._fitted = True

    def predict_proba(self, V: np.ndarray) -> np.ndarray:
        """
        V: (N,5) raw belief vectors.
        Return: P(correct) = \hat p_c ∈ (0,1).
        """
        if not self._fitted:
            raise RuntimeError("LogicConstrainedCalibrator not fitted.")

        self.model.eval()
        with torch.no_grad():
            v = torch.from_numpy(V.astype(np.float32)).to(self.device)
            logits_hat = self.model(v)              # (N,5)
            p_hat = torch.sigmoid(logits_hat)       # (N,5)
            p_c = p_hat[:, 0]                       # (N,)
            return p_c.cpu().numpy()

    def predict_beliefs(self, V: np.ndarray) -> np.ndarray:
        """
        Optional helper: trả về full \hat v (N,5) sau khi calibrate.
        """
        if not self._fitted:
            raise RuntimeError("LogicConstrainedCalibrator not fitted.")

        self.model.eval()
        with torch.no_grad():
            v = torch.from_numpy(V.astype(np.float32)).to(self.device)
            logits_hat = self.model(v)
            p_hat = torch.sigmoid(logits_hat)
            return p_hat.cpu().numpy()
