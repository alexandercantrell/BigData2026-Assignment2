import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch as ch
from constants import DATA, OUT, TARGETS, DATA_END, TEST_INPUT
from data import load_tables, build_graph
from model import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',default=None)
    parser.add_argument('--input',type=Path, default=TEST_INPUT)
    parser.add_argument('--name',default="model")
    return parser.parse_args()

def main():
    args = parse_args()
    ckpt = ch.load(OUT / f"{args.name}.ckpt")
    tables = load_tables(DATA)
    graph = build_graph(tables,DATA_END,ckpt['features'],ckpt.get('reverse',True))
    model = Model(graph,ckpt['hidden'],ckpt['layers'],len(TARGETS),ckpt.get('aggr','all'))
    model.load_state_dict(ckpt["model"])
    model.eval()

    test = pd.read_csv(args.input)
    row_of = pd.Series(tables['Users'].index,index=tables["Users"]["github_user_id"].astype("int64"))
    rows = test["UserID"].map(row_of)
    seen = rows.notna().to_numpy()

    pred = np.zeros((len(test),len(TARGETS)),dtype=int)
    with ch.no_grad():
        out = model(graph,ch.tensor(rows[seen].astype('int64').to_numpy())).sigmoid().numpy()
    pred[seen] = (out>=np.array(ckpt['thresholds'])).astype(int)

    out = pd.concat([test[["UserID"]],pd.DataFrame(pred,columns=TARGETS)],axis=1)
    stem = f"output_{args.id}" if args.id else "output"
    path = OUT/(f"{stem}.csv" if args.name == "model" else f"{stem}_{args.name}.csv")
    OUT.mkdir(exist_ok=True)
    out.to_csv(path,index=False)
    print(f"{args.name} (epoch {ckpt['epoch']}, features: constant + {ckpt['features']}): wrote {path}")
    print(f"{len(out)} users ({int(seen.sum())} seen in the data, {int((~seen).sum())} all-zero); "
          f"positives per activity: {out[TARGETS].sum().to_dict()}")

if __name__ == "__main__":
    main()