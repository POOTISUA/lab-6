"""Модель Net, формула кількості параметрів та завантаження MedMNIST (ЛР6, Варіант 24: Д6 OCTMNIST, Ш3)."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import TensorDataset

DATASETS = {
    "Д1": ("dermamnist", 7, 3),
    "Д2": ("bloodmnist", 8, 3),
    "Д3": ("organamnist", 11, 1),
    "Д4": ("organcmnist", 11, 1),
    "Д5": ("pathmnist", 9, 3),
    "Д6": ("octmnist", 4, 1),
}

WIDTHS = {
    "Ш1": (16, 32, 128),
    "Ш2": (32, 64, 256),
    "Ш3": (64, 128, 512),
}

# Параметри варіанта 24
VARIANT = 24
DS_KEY = "Д6"
WIDTH_KEY = "Ш3"
P_MAX = 4
B_GLOBAL = 192
B_LOCAL = B_GLOBAL // P_MAX  # 48
K_C = 15
S_F = 37
N_TRAIN = 7000
N_EP = 3
LR0 = 0.02
MOMENTUM = 0.9


def load_split(path: str, split: str, n: int = None) -> TensorDataset:
    """Розбиття split ('train' або 'test') файлу .npz як TensorDataset."""
    d = np.load(path)
    x = d[f"{split}_images"][:n]
    y = d[f"{split}_labels"][:n].reshape(-1)
    if x.ndim == 3:
        x = x[:, None, :, :]
    else:
        x = x.transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = (x / 255.0 - 0.5) / 0.5
    return TensorDataset(x, torch.from_numpy(y.astype(np.int64)))


class Net(nn.Module):
    def __init__(self, C: int, c1: int, c2: int, h: int, K: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * c2, h),
            nn.ReLU(),
            nn.Linear(h, K),
        )

    def forward(self, x):
        return self.head(self.features(x))


def n_params_formula(C: int, c1: int, c2: int, h: int, K: int) -> int:
    return 9 * C * c1 + c1 + 9 * c1 * c2 + c2 + 49 * c2 * h + h + h * K + K


def evaluate_model(model: nn.Module, test_loader) -> tuple:
    """Обчислює ACC та macro-average OvR AUC на тестовій вибірці."""
    model.eval()
    all_probs = []
    all_y = []
    correct = 0
    total = 0
    non_finite_logits = False
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            if not torch.isfinite(logits).all():
                non_finite_logits = True
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
            all_probs.append(probs.numpy())
            all_y.append(yb.numpy())
    y_true = np.concatenate(all_y, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    acc = float("nan") if non_finite_logits else (correct / total)
    if not np.isfinite(y_prob).all():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    return acc, auc
