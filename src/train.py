import argparse
import time
import json
import numpy as np
import torch as ch
from tqdm import tqdm
from metrics import tune_thresholds, score
from constants import DATA, OUT, SEED, TARGETS, LR, TRAIN_WEEKS, VAL_WEEKS, HIDDEN_DIM, GNN_LAYERS, EVAL_EVERY, EPOCHS, FEATURE_GROUPS
from data import load_tables, weekly_sample, copy_last_week
from model import Model

def evaluate(model,data,targets):
    model.eval()
    with ch.no_grad():
        out = np.concatenate([model(graph,users).sigmoid().numpy() for graph,users, _ in data])
    thresh = tune_thresholds(out,targets)
    return score((out>=thresh).astype(int),targets),thresh

def train_epoch(model,opt,loss_fn,data,epoch):
    model.train()
    total_loss=0.0
    n=0
    bar = tqdm(data, desc=f"epoch {epoch:2d}", leave=False)
    for graph, users, targets in bar:
        opt.zero_grad()
        loss = loss_fn(model(graph,users),targets)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n+=1
        bar.set_postfix(loss=f"{total_loss/n:.4f}")
    return total_loss/n

def train(model, train_data, val_data, features,arch, ckpt_path):
    opt = ch.optim.Adam(model.parameters(),lr=LR)
    loss_fn = ch.nn.BCEWithLogitsLoss()
    best = {'score':-1.0}
    val_targets = ch.cat([tgt for *_, tgt in val_data]).numpy()
    start = time.time()
    for epoch in range(1,EPOCHS+1):
        loss = train_epoch(model,opt,loss_fn, train_data, epoch)
        if epoch % EVAL_EVERY and epoch != EPOCHS:
            continue
        s, thresh = evaluate(model, val_data, val_targets)
        print(f"epoch {epoch:2d}  loss {loss:.4f}  val {s['score']:.2%}")
        if s['score'] > best['score']:
            best = {"score": s['score'], 'epoch':epoch,'thresholds':thresh,'val':s}
            save_checkpoint(ckpt_path,model,opt,epoch,features,arch,thresh,s["score"])
            print(f"Saved new best: {best['score']:.2%}")
    end = time.time()
    return best, end-start
        

def save_checkpoint(path, model, opt, epoch, features,arch, thresholds, val_score):
    ch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": epoch, "features": features, "thresholds": thresholds.tolist(),
                "hidden": HIDDEN_DIM, **arch, "val_score": val_score}, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--bools",action='store_true')
    parser.add_argument('--present',action='store_true')
    parser.add_argument('--category',action='store_true')
    parser.add_argument('--name',default='model')
    parser.add_argument('--aggr',choices=['all','mean'],default='all')
    parser.add_argument('--layers',type=int,default=GNN_LAYERS)
    parser.add_argument('--no-reverse',action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    features = [group for group in FEATURE_GROUPS if getattr(args,group)]
    arch = {'layers':args.layers,'aggr':args.aggr,'reverse':not args.no_reverse}
    print(f"{args.name}: features = constant + {features or 'nothing'}")

    ch.manual_seed(SEED)
    np.random.seed(SEED)
    tables = load_tables(DATA)
    train_data = [weekly_sample(tables,week,features,arch['reverse']) for week in TRAIN_WEEKS]
    val_data = [weekly_sample(tables,week,features,arch['reverse']) for week in VAL_WEEKS]
    val_targets = ch.cat([tgt for *_,tgt in val_data]).numpy()
    print(f"train weeks {[f'{w:%m-%d}' for w in TRAIN_WEEKS]}: {sum(len(u) for _, u, _ in train_data):,} examples")
    print(f"val weeks   {[f'{w:%m-%d}' for w in VAL_WEEKS]}: {len(val_targets):,} examples")

    OUT.mkdir(exist_ok=True)
    ckpt_path = OUT/f"{args.name}.ckpt"
    model = Model(train_data[0][0], HIDDEN_DIM, arch['layers'], len(TARGETS),arch['aggr'])
    best, train_time = train(model, train_data,val_data,features,arch,ckpt_path)
    base = score(np.concatenate([copy_last_week(tables,week) for week in VAL_WEEKS]), val_targets)

    results = {"name":args.name,'features':features,'best_epoch':best["epoch"],"thresholds":best["thresholds"].round(2).tolist(),
               **best["val"], "copy_last_week":base, 'train_time':train_time,**arch}
    (OUT/f"{args.name}.json").write_text(json.dumps(results,indent=2))
    print(f"Best epoch {best['epoch']}: Val Score {best['score']:.2%}  Macro {best['val']['macro_f1']:.2%}  "
          f"Micro {best['val']['micro_f1']:.2%}")
    print("Per Activity:", {k: f"{v:.1%}" for k, v in best["val"]["per_activity"].items()})
    print(f"Copy-Last-Week on the same examples: Score {base['score']:.2%}")
    print(f"Best Checkpoint: {ckpt_path}")
    print(f"Train time: {train_time}")

if __name__ == "__main__":
    main()