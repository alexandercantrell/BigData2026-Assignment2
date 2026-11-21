import numpy as np
from constants import TARGETS

# F1 and Score follow the course's evaluate.py (scores()), so validation uses exactly the official metric.
def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0

def score(pred, truth):
    tp = ((pred == 1) & (truth == 1)).sum(0)
    fp = ((pred == 1) & (truth == 0)).sum(0)
    fn = ((pred == 0) & (truth == 1)).sum(0)
    per = {k: float(f1(*c)) for k, c in zip(TARGETS, zip(tp, fp, fn))}
    macro = float(np.mean(list(per.values())))
    micro = float(f1(tp.sum(), fp.sum(), fn.sum()))
    return {"score": (macro + micro) / 2, "macro_f1": macro, "micro_f1": micro, "per_activity": per}

def tune_thresholds(p, y, grid=np.linspace(0.02, 0.98, 49)):
    best = []
    for j in range(y.shape[1]):
        f1s = [f1(((p[:, j] >= c) & (y[:, j] == 1)).sum(), ((p[:, j] >= c) & (y[:, j] == 0)).sum(),
                  ((p[:, j] < c) & (y[:, j] == 1)).sum()) for c in grid]
        best.append(grid[int(np.argmax(f1s))])
    return np.array(best)