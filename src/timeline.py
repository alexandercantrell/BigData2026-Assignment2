import matplotlib.pyplot as plt
from constants import ROOT, DATA_START, TEST_WEEK, TRAIN_WEEKS, VAL_WEEKS, GAP, WEEK

weeks = [("train", w) for w in TRAIN_WEEKS] + [("val", w) for w in VAL_WEEKS] + [("test", TEST_WEEK)]
colors = {"train": "blue", "val": "orange", "test": "red"}

fig, ax = plt.subplots(figsize=(10, 4))
for row, (kind, week) in enumerate(weeks):
    ax.barh(row, week - GAP - DATA_START, left=DATA_START, color="lightblue")
    ax.barh(row, GAP, left=week - GAP, color="lightgray")
    ax.barh(row, WEEK, left=week, color=colors[kind])

ax.set_yticks(range(len(weeks)), [f"{kind} {week:%b %d}" for kind, week in weeks])
ax.invert_yaxis()
ax.set_title("Data seen (light blue) -> 7-day gap (gray) -> week predicted")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "timeline.png")