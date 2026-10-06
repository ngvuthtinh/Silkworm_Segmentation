"""Chia train/valid/test theo NHÓM CẢNH, sạch rò rỉ (SIFT ≥8 cùng lớp, hoặc pixel ≥0.85)."""
import json, re, random, collections, sys, numpy as np
SCR = sys.argv[1]
D = json.load(open(f"{SCR}/scene_groups_th10.json")); items = [tuple(x) for x in D["items"]]; G = D["groups"]; N = len(items)
H = np.load(f"{SCR}/H96.npy"); C = np.load(f"{SCR}/cand_pairs.npy"); R = np.load(f"{SCR}/cand_inl.npy")
bad = json.load(open(f"{SCR}/bad_masks_v3.json")); ok = json.load(open(f"{SCR}/rescued_ok.json"))
drop = {(sp, s) for sp in bad for s in set(bad[sp]) - set(ok[sp])}
clsv = np.array([1 if st.lower().startswith("healthy") else 0 for _, st in items])
gid = np.zeros(N, int)
for gi, g in enumerate(G): gid[g] = gi
base = lambda i: re.sub(r"_jpg\.rf\..*$", "", items[i][1])
link = collections.defaultdict(set)
for (a, b), r in zip(C, R):
    if r >= 8 and clsv[a] == clsv[b] and gid[a] != gid[b]:
        link[gid[a]].add(gid[b]); link[gid[b]].add(gid[a])
def one_copy(g):
    seen, out = set(), []
    for i in sorted(g, key=lambda i: items[i][1]):
        if items[i] in drop or base(i) in seen: continue
        seen.add(base(i)); out.append(i)
    return out
TARGET, CAP = 250, 8
rng = random.Random(42); order = list(range(len(G))); rng.shuffle(order)
assign = {gi: "train" for gi in range(len(G))}
def fill():
    for split in ["valid", "test"]:
        for c in [0, 1]:
            n = sum(min(len(one_copy(G[gi])), CAP) for gi, s in assign.items() if s == split and clsv[G[gi][0]] == c)
            for gi in order:
                if n >= TARGET: break
                if assign[gi] != "train" or gi in banned or clsv[G[gi][0]] != c: continue
                m = one_copy(G[gi])
                if not (1 <= len(m) <= 40) or link[gi]: continue      # nhóm có BẤT KỲ liên kết mơ hồ nào → ở lại train
                assign[gi] = split; n += min(len(m), CAP)
banned = set()
for it in range(10):
    fill()
    tr = np.array([i for gi, s in assign.items() if s == "train" for i in G[gi] if items[i] not in drop])
    ev = [(gi, i) for gi, s in assign.items() if s != "train" for i in G[gi] if items[i] not in drop]
    sim = (H[[i for _, i in ev]] @ H[tr].T).max(1)
    bad_g = {gi for (gi, _), v in zip(ev, sim) if v >= 0.85}
    for gi in bad_g: assign[gi] = "train"; banned.add(gi)
    print(f"vòng {it}: trả {len(bad_g)} nhóm có ảnh giống train (pixel ≥0.85) về train", flush=True)
    if not bad_g: break
manifest, unused = [], 0
for gi, g in enumerate(G):
    s = assign[gi]
    keep = [i for i in g if items[i] not in drop] if s == "train" else rng.sample(one_copy(g), min(CAP, len(one_copy(g))))
    unused += len(g) - len(keep)
    for i in keep:
        manifest.append({"stem": items[i][1], "src_split": items[i][0], "split": s, "group": int(gi), "cls": int(clsv[i])})
M = collections.defaultdict(list)
for r in manifest: M[r["split"]].append(r)
for s in ["train", "valid", "test"]:
    c = collections.Counter(r["cls"] for r in M[s])
    print(f"{s:5s}: {len(M[s]):5d} ảnh | {len({r['group'] for r in M[s]}):4d} cảnh | Grasserie {c[0]} / Healthy {c[1]}")
print("không dùng:", unused, "| tổng:", len(manifest) + unused, "=", N)
json.dump(manifest, open(f"{SCR}/split_manifest_v3.json", "w"))
idx = {it: i for i, it in enumerate(items)}; sp = {idx[(r["src_split"], r["stem"])]: r["split"] for r in manifest}
for s in ["valid", "test"]:
    ids = [i for i, v in sp.items() if v == s]; tr = [i for i, v in sp.items() if v == "train"]; oth = [i for i, v in sp.items() if v not in (s, "train")]
    best = collections.defaultdict(int)
    for (a, b), r in zip(C, R):
        if sp.get(a) == s and sp.get(b) not in (None, s): best[a] = max(best[a], r)
        if sp.get(b) == s and sp.get(a) not in (None, s): best[b] = max(best[b], r)
    v = np.array([best.get(i, 0) for i in ids])
    print(f"  {s}: khớp SIFT với split khác ≥8: {(v>=8).sum()} | ≥20: {(v>=20).sum()} | pixel với train ≥0.85: {((H[ids]@H[tr].T).max(1)>=0.85).sum()} | pixel với split kia ≥0.85: {((H[ids]@H[oth].T).max(1)>=0.85).sum()}")
