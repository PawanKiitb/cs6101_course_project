import sys, time, torch
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
torch.set_num_threads(1)

K = int(sys.argv[1])
P = {10: (1, 0.5, 256), 100: (2, 0.45, 1024), 1000: (4, 0.4, 4096)}[K]  # verify vs searcher.py defaults
IDX = "/home/pawankmeena197/plaid/native"
CKPT = "/home/pawankmeena197/plaid/colbertv2.0"
COLL = "/home/pawankmeena197/plaid/msmarco/collection.tsv"
QRY  = "/home/pawankmeena197/plaid/msmarco/queries.dev.small.tsv"

if __name__ == "__main__":
    with Run().context(RunConfig(nranks=1, experiment="plaid")):
        s = Searcher(index=IDX, checkpoint=CKPT, collection=COLL, config=ColBERTConfig())
        s.configure(ncells=P[0], centroid_score_threshold=P[1], ndocs=P[2])
        qs = list(Queries(QRY).items())
        enc, ret, out = [], [], []
        for i, (qid, text) in enumerate(qs):
            t0 = time.perf_counter(); Q = s.encode(text); t1 = time.perf_counter()
            pids, ranks, scores = s.dense_search(Q, k=K); t2 = time.perf_counter()
            if i >= 50:                      # skip warm-up
                enc.append(t1 - t0); ret.append(t2 - t1)
            out += [f"{qid}\t{p}\t{r}\t{sc}" for p, r, sc in zip(pids, ranks, scores)]
        open(f"plaid_k{K}.tsv", "w").write("\n".join(out) + "\n")
        n = len(ret)
        print(f"k={K} queries timed={n} encode_ms={1000*sum(enc)/n:.1f} "
              f"search_ms={1000*sum(ret)/n:.1f} total_ms={1000*(sum(enc)+sum(ret))/n:.1f}")
