import pandas as pd
from constants import SCHEMA, ACTIVITY, TARGETS, FEATURES, GAP, WEEK
import numpy as np
import torch as ch
from torch_geometric.data import HeteroData

def load_tables(dir):
    return {name: pd.read_parquet(dir / f"{name}.parquet") for name in SCHEMA}

def snapshot(tables, time):
    out = {}
    for name, df in tables.items():
        time_col = SCHEMA[name]["time"]
        later_cols = SCHEMA[name].get("known_later", {})
        df = df[df[time_col] < time].copy()
        for later_col, cols in later_cols.items():
            df.loc[~(df[later_col] < time), cols] = None
        out[name] = df
    return out

def events(tables, target):
    table, user_col, time_col = ACTIVITY[target]
    df = tables[table]
    if target == 'open_issue':
        df = df[~df['is_pull_request'].fillna(False)]
    return df[[user_col,time_col]].dropna().set_axis(['user','time'],axis=1)

def make_labels(tables, users, start, end):
    y = pd.DataFrame(0,index=pd.Index(users,name='user_id'),columns=TARGETS)
    for target in TARGETS:
        ev = events(tables,target)
        did = ev.loc[(ev['time']>=start) & (ev['time'] < end),'user'].astype('int64').unique()
        y.loc[y.index.isin(did),target]=1
    return y

def select_users(tables, time):
    users = snapshot(tables, time)["Users"]
    return users.index[users['is_cohort'].fillna(False).to_numpy()].tolist()

def node_features(name, df, time, features):
    table = FEATURES.get(name, {})
    cols = [np.ones(len(df))]
    if "bools" in features:
        cols += [df[c].fillna(False).astype(float).to_numpy() for c in SCHEMA[name].get("bool", [])]
    if "present" in features:
        cols += [df[c].notna().astype(float).to_numpy() for c in table.get("present", [])]
    if "category" in features:
        for col, values in table.get("category", {}).items():
            cols += [(df[col] == v).fillna(False).astype(float).to_numpy() for v in values]
    if "time" in features:
        age_days = (time - df[SCHEMA[name]["time"]]).dt.total_seconds().to_numpy() / (60 * 60 * 24)
        cols.append(np.log1p(age_days))
    return ch.tensor(np.column_stack(cols), dtype=ch.float)

def build_graph(tables, time, features,reverse=True):
    snap = snapshot(tables, time)
    data = HeteroData()
    for name, df in snap.items():
        data[name].x = node_features(name, df, time, features)
    for src, spec in SCHEMA.items():
        for col, dst in spec.get("fks", {}).items():
            fk = snap[src][col]
            valid = (fk.notna() & (fk < len(snap[dst]))).fillna(False).to_numpy()
            s = ch.from_numpy(np.flatnonzero(valid))
            d = ch.tensor(fk[valid].astype("int64").to_numpy())
            data[src, col, dst].edge_index = ch.stack([s, d])
            if reverse:
                data[dst, f"rev_{col}", src].edge_index = ch.stack([d, s])
    return data

def weekly_sample(tables, week, features, reverse=True):
    known = week-GAP
    users = select_users(tables,known)
    graph = build_graph(tables,known,features, reverse=reverse)
    targets = make_labels(tables,users,week,week+WEEK)
    return graph, ch.tensor(users), ch.tensor(targets.to_numpy(),dtype=ch.float) 

def copy_last_week(tables,week):
    known = week-GAP
    users=select_users(tables,known)
    return make_labels(snapshot(tables,known),users,known-WEEK,known).to_numpy()