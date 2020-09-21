import json

all_relations = set()
with open('normal/train.txt', 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        if sample['relation'] not in all_relations:
            all_relations.add(sample['relation'])
rel2id = {rel: idx for idx, rel in enumerate(all_relations)}
with open('rel2id.json', 'w') as f:
    json.dump(rel2id, f)
