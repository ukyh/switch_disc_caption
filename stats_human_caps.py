import json
import os
import pathlib
from collections import Counter

import random
random.seed(123)


DATASET = "data/dataset_coco.json"
COCOTALK = "data/cocotalk.json"
OUTDIR = "processed_human_caps"
CLIP_CAND = os.path.join(OUTDIR, "human_cap_cands_test.json")
CLIP_REF = os.path.join(OUTDIR, "human_cap_refs_test.json")
BERTPP = os.path.join(OUTDIR, "human_cap_bertpp_test.json")
VSEPP = os.path.join(OUTDIR, "human_cap_vsepp_test.json")
N = 4


def n_gram(target, n):
    assert n > 0
    return [" ".join(target[idx:idx + n]) for idx in range(len(target) - n + 1)]


dataset = json.load(open(DATASET))
cocotalk = json.load(open(COCOTALK))


# Split
train = {}
val = {}
test = {}
for split_dict in cocotalk["images"]:
    imgid = split_dict["id"]
    if split_dict["split"] == "test":
        test[imgid] = []
    elif split_dict["split"] == "val":
        val[imgid] = []
    else:   # train or restval
        train[imgid] = []
assert len(test) == len(val) == 5000

# Vocab
word_to_ix = {}
for ix, word in cocotalk["ix_to_word"].items():
    word_to_ix[word] = int(ix)

# Convert
cands = {}  # {"fname":"cap", ...}
refs = {}   # {"fname":["cap1", ...], ...}
cands_vsepp = []    # [{"imgid":imgid, "caption":"cap"}, ...]
for img_dict in dataset["images"]:
    imgid = img_dict["cocoid"]
    if imgid in test:   # test split only
        imgfile = img_dict["filename"] # e.g., "COCO_val2014_000000391895.jpg"
        imgfile_stem = pathlib.Path(imgfile).stem   # e.g., "COCO_val2014_000000391895"
        rand_idx = random.randint(0, len(img_dict["sentences"]) - 1)
        for i in range(len(img_dict["sentences"])):
            _cap = img_dict["sentences"][i]["tokens"]   # ["tokens", "in", "sentence"]
            if i == rand_idx:  # random cand
                cap = " ".join(_cap)
                cands[imgfile_stem] = cap
                cands_vsepp.append({"image_id":imgid, "caption":cap})
            else:
                cap = " ".join(_cap)    # ref is free from <unk>
                if imgfile_stem in refs:
                    refs[imgfile_stem].append(cap)
                else:
                    refs[imgfile_stem] = [cap]
    else:
        pass
assert len(cands) == len(refs) == len(cands_vsepp)

with open(CLIP_CAND, "w") as clip_cand_out:
    json.dump(cands, clip_cand_out, indent=4)
print("Created {}".format(CLIP_CAND))

with open(CLIP_REF, "w") as clip_ref_out:
    json.dump(refs, clip_ref_out, indent=4)
print("Created {}".format(CLIP_REF))

bertpp = {}
for i, (fname, cand_cap) in enumerate(cands.items()):
    bertpp[i] = {"refs":refs[fname], "cand":[cand_cap]} # {ix:{"refs":["cap1", ...], "cand":["cap"]}, ...}
with open(BERTPP, "w") as bertpp_out:
    json.dump(bertpp, bertpp_out, indent=4)
print("Created {}".format(BERTPP))

with open(VSEPP, "w") as vsepp_out:
    json.dump(cands_vsepp, vsepp_out, indent=4)
print("Created {}".format(VSEPP))

## Stats
va = []
alen = []
sa = []
for fname, cand_cap in cands.items():
    va.extend(cand_cap.split())
    alen.append(len(cand_cap.split()))
    sa.append(cand_cap)
assert len(alen) == len(sa) == 5000
va = set(va)
sa = set(sa)
print("## Stats on human cands")
print("avg length of human cands: {}".format(sum(alen) / len(alen)))
print("vocab size of human cands: {}".format(len(va)))
print("unique sentences in human cands: {}".format(len(sa)))

train_freq = []
for img_dict in dataset["images"]:
    imgid = img_dict["cocoid"]
    if (imgid not in val) and (imgid not in test):
        cap_list = []
        for cap_dict in img_dict["sentences"]:
            cap_list.extend(cap_dict["tokens"])
        train_freq.extend(cap_list)

# Sort
rank_dict = {}
train_count = Counter(train_freq)
_v = -1
_i = 0
for i, (k, v) in enumerate(train_count.most_common(None)):
    if v == _v:
        rank_dict[k] = _i
    else:
        rank_dict[k] = i
        _i = i
        _v = v
oov_rank = _i + 1

# OOR stats
not_in_words = []
not_in_rank = []
in_words = []
in_rank = []
oov_count = 0
for fname, cand_cap in cands.items():
    _ref_caps = refs[fname]
    ref_caps = sum([rcap.split() for rcap in _ref_caps], [])
    ref_words = set(ref_caps)
    for word in cand_cap.split():
        if word not in ref_words:
            not_in_words.append(word)
            if word in rank_dict:
                not_in_rank.append(rank_dict[word])
            else:
                not_in_rank.append(oov_rank)
                oov_count += 1
        else:
            in_words.append(word)
            if word in rank_dict:
                in_rank.append(rank_dict[word])
            else:
                in_rank.append(oov_rank)
                oov_count += 1
assert len(not_in_words) == len(not_in_rank)
assert len(in_words) == len(in_rank)
print("The number of OOV word: {}".format(oov_count))

print("## Underrated Stats on human cands")
print("The number of OOR words: {}".format(len(not_in_words)))
print("The average rank of OOR words: {}".format(sum(not_in_rank) / len(not_in_rank)))
print("The number of in-ref words: {}".format(len(in_words)))
print("The average rank of in-ref words: {}".format(sum(in_rank) / len(in_rank)))

# Repetition stats
rep_list = []
for fname, cand_cap in cands.items():
    n_rep_list = []
    for n in range(1, N + 1):
        _cand_cap = cand_cap.split()
        _ngrams = n_gram(_cand_cap, n)
        _ngrams_set = set(_ngrams)
        n_rep_num = len(_ngrams) - len(_ngrams_set)
        n_rep_list.append(n_rep_num / len(_ngrams))     # avg in each n-gram
    assert len(n_rep_list) == N
    rep_list.append(sum(n_rep_list) / len(n_rep_list))  # avg in each sentence
assert len(rep_list) == 5000

print("## Repetition stats of human cands")
print("avg percentage of [1,4]-gram repetition in human cands: {}".format(sum(rep_list) / len(rep_list)))


print("Finished process")
