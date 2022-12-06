import json
import sys
from collections import Counter


DATASET = "data/dataset_coco.json"
COCOTALK = "data/cocotalk.json"
OUTPUT = sys.argv[1]
assert "test" in OUTPUT, "Currently, this stats supports test only"


dataset = json.load(open(DATASET))
cocotalk = json.load(open(COCOTALK))
output = json.load(open(OUTPUT))


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
assert len(test) == len(output["imgToEval"])

# Vocab
word_to_ix = {}
for ix, word in cocotalk["ix_to_word"].items():
    word_to_ix[word] = int(ix)

# Convert
train_freq = []
for img_dict in dataset["images"]:
    imgid = img_dict["cocoid"]
    cap_list = []
    for cap_dict in img_dict["sentences"]:
        cap_list.extend(cap_dict["tokens"])
    if imgid in test:
        cap_set = set(cap_list)
        test[imgid] = cap_set
    elif imgid in val:
        pass
    else:   # train or restval
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

# Stats
not_in_words = []
not_in_rank = []
in_words = []
in_rank = []
oov_count = 0
for imgid in output["imgToEval"]:   # str(image_id)
    cap = output["imgToEval"][imgid]["caption"].split()
    for word in cap:
        if word not in test[int(imgid)]:
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

print("## Underrated Stats on {}".format(OUTPUT))
print("The number of OOR words: {}".format(len(not_in_words)))
print("The average rank of OOR words: {}".format(sum(not_in_rank) / len(not_in_rank)))
print("The number of in-ref words: {}".format(len(in_words)))
print("The average rank in-ref words: {}".format(sum(in_rank) / len(in_rank)))
print("Finished process")
