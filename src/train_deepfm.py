"""DeepFM CTR model (PyTorch). Hyperparameters and early stopping use VALID only; TEST is predicted once."""
import argparse, copy, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import load, build_vocab, encode, save_result

SEED = 42
EMB_DIM = 16
HIDDEN = [256, 128]
DROPOUT = 0.2
LR = 1e-3
BATCH = 4096
MAX_EPOCHS = 3


class NonFiniteLoss(RuntimeError):
    pass


class DeepFM(nn.Module):
    def __init__(self, field_sizes, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.n_fields = len(field_sizes)
        self.emb_dim = emb_dim
        offsets = np.concatenate([[0], np.cumsum(field_sizes)[:-1]]).astype("int64")
        self.register_buffer("offsets", torch.from_numpy(offsets))
        total = int(sum(field_sizes))
        self.emb = nn.Embedding(total, emb_dim)      # shared by FM and deep parts
        self.lin = nn.Embedding(total, 1)            # first-order term
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.lin.weight)
        layers, d = [], self.n_fields * emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        idx = x + self.offsets
        e = self.emb(idx)                                  # [B, F, K]
        first = self.lin(idx).sum(dim=(1, 2))              # [B]
        sum_sq = e.sum(dim=1).pow(2)
        sq_sum = e.pow(2).sum(dim=1)
        second = 0.5 * (sum_sq - sq_sum).sum(dim=1)        # [B]
        deep = self.mlp(e.flatten(1)).squeeze(1)           # [B]
        return first + second + deep + self.bias           # logits


@torch.no_grad()
def predict(model, X, device, bs=16384):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out).astype("float64")


def train(model, Xtr, ytr, Xva, yva, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    best_auc, best_epoch, best_state, epochs_run = -1.0, 0, None, 0
    n = len(Xtr)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n)
        t0, cnt = time.time(), 0
        tot = torch.zeros((), device=device)  # accumulated on device, synced once per epoch
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(ytr[b]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)
            cnt += len(b)
        tot_loss = float(tot.item()) / cnt
        if not np.isfinite(tot_loss):
            raise NonFiniteLoss(f"non-finite train loss at epoch {ep} on {device}")
        pva = predict(model, Xva, device)
        auc = roc_auc_score(yva, pva)
        epochs_run = ep
        print(f"epoch {ep}: train_loss={tot_loss:.4f} valid_auc={auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            break  # patience 0 by design: stop at the first valid AUC drop, keep best epoch
    model.load_state_dict(best_state)
    return epochs_run, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t_script = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tr, va, te = load("train", args.limit), load("valid", args.limit), load("test", args.limit)
    vocab = build_vocab(tr)
    Xtr, Xva, Xte = encode(tr, vocab), encode(va, vocab), encode(te, vocab)
    ytr = tr["label"].to_numpy().astype("float32")
    yva = va["label"].to_numpy()
    yte = te["label"].to_numpy()
    del tr, va, te
    field_sizes = [len(vocab[c]) + 1 for c in vocab]  # +1 for the unseen bucket 0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t_wall = time.time()  # wall clock including model build, training and prediction
    t_start = time.time()
    model = DeepFM(field_sizes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params} vocab_total={sum(field_sizes)}", flush=True)
    try:
        model.to(device)
        epochs_run, best_epoch = train(model, Xtr, ytr, Xva, yva, device)
    except RuntimeError as e:  # MPS device errors and NonFiniteLoss; other bugs propagate
        if device == "cpu":
            raise
        print(f"WARNING: mps run failed ({type(e).__name__}: {e}); retraining on cpu", flush=True)
        device = "cpu"
        torch.manual_seed(SEED)
        model = DeepFM(field_sizes).to(device)
        epochs_run, best_epoch = train(model, Xtr, ytr, Xva, yva, device)
    train_seconds = time.time() - t_start

    # TEST touched once, with best-by-VALID weights (already loaded). If the device call
    # fails, move the trained model to cpu rather than losing it.
    try:
        p_test = predict(model, Xte, device)
    except RuntimeError as e:
        print(f"WARNING: test predict failed on {device} ({e}); using cpu", flush=True)
        device = "cpu"
        model = model.to("cpu")
        p_test = predict(model, Xte, device)
    extra = {"epochs_run": epochs_run, "best_epoch": best_epoch,
             "train_seconds": round(train_seconds, 1),  # model build + training + valid eval, excludes data load/encode
             "wall_seconds": round(time.time() - t_wall, 1), "total_seconds": round(time.time() - t_script, 1), "device": device, "n_params": int(n_params)}
    save_result("deepfm", yte, p_test, extra)


if __name__ == "__main__":
    main()
