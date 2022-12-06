import json
import sys


aname = sys.argv[1]


a = open(aname)
a = json.load(a)
N = 4

def n_gram(target, n):
    assert n > 0
    return [" ".join(target[idx:idx + n]) for idx in range(len(target) - n + 1)]

va = []
sa = []
alen = []
rep_list = []
if "imgToEval" in a:
    for key in a["imgToEval"]:
        acap = a["imgToEval"][key]["caption"]
        acap = acap.split()
        n_rep_list = []
        for n in range(1, N + 1):
            _ngrams = n_gram(acap, n)
            _ngrams_set = set(_ngrams)
            n_rep_num = len(_ngrams) - len(_ngrams_set)
            n_rep_list.append(n_rep_num / len(_ngrams))     # avg in each n-gram
        assert len(n_rep_list) == N
        rep_list.append(sum(n_rep_list) / len(n_rep_list))  # avg in each sentence
        va.extend(acap)
        alen.append(len(acap))
        sa.append(" ".join(acap))
else:   # Oscar
    for item in a:
        acap = item["caption"].rstrip(".")  # Oscar outputs are not tokenized
        acap = acap.split()
        n_rep_list = []
        for n in range(1, N + 1):
            _ngrams = n_gram(acap, n)
            _ngrams_set = set(_ngrams)
            n_rep_num = len(_ngrams) - len(_ngrams_set)
            n_rep_list.append(n_rep_num / len(_ngrams))     # avg in each n-gram
        assert len(n_rep_list) == N
        rep_list.append(sum(n_rep_list) / len(n_rep_list))  # avg in each sentence
        va.extend(acap)
        alen.append(len(acap))
        sa.append(" ".join(acap))

assert len(alen) == len(rep_list) == len(sa)
va = set(va)
sa = set(sa)

print("## Repetition stats of {}".format(aname))
print("avg percentage of [1,4]-gram repetition in {}: {}".format(aname, sum(rep_list) / len(rep_list)))
print("avg length of {}: {}".format(aname, sum(alen) / len(alen)))
print("vocab size of {}: {}".format(aname, len(va)))
print("unique sentences in {}: {}".format(aname, len(sa)))
