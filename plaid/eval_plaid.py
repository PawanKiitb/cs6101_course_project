import sys
from collections import defaultdict

ranking_file, qrels_file = sys.argv[1], sys.argv[2]

qrels = defaultdict(set)
for line in open(qrels_file):
    qid, _, pid, rel = line.split()
    if int(rel) > 0:
        qrels[int(qid)].add(int(pid))

ranks = defaultdict(list)
for line in open(ranking_file):
    f = line.strip().split("\t") if "\t" in line else line.split()
    ranks[int(f[0])].append((int(f[2]), int(f[1])))  # (rank, pid)

qids = list(qrels)
mrr, rec = 0.0, defaultdict(float)
cuts = [10, 50, 100, 200, 1000]
for q in qids:
    ordered = [pid for _, pid in sorted(ranks.get(q, []))]
    for i, pid in enumerate(ordered[:10]):
        if pid in qrels[q]:
            mrr += 1 / (i + 1)
            break
    for c in cuts:
        rec[c] += len(set(ordered[:c]) & qrels[q]) / len(qrels[q])

n = len(qids)
print(f"queries: {n}")
print(f"MRR@10: {mrr/n:.4f}")
for c in cuts:
    print(f"Recall@{c}: {rec[c]/n:.4f}")
