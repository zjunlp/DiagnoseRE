import sys
import os
import re
import json
import logging
import OpenAttack
import numpy as np
import opennre
import torch
import argparse
from opennre.model import SoftmaxNN
from OpenAttack.utils.dataset import Dataset, DataInstance
from tqdm import tqdm
import logging
import tensorflow as tf


class REClassifier(OpenAttack.Classifier):
    def __init__(self, max_seq_len, model_path, rel2id, id2rel, device):
        # Some basic settings
        root_path = '../'
        sys.path.append(root_path)
        self.device = device

        # Load original BERT model
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=max_seq_len,
            pretrain_path=os.path.join(
                root_path, 'pretrain/bert-base-uncased'),
            mask_entity=False)
        self.model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path)['state_dict'])
        self.max_seq_len = max_seq_len
        self.id2rel = id2rel
        self.tokenizer = self.model.sentence_encoder.tokenizer
        self.current_label = -1

    def tokenize_word_list(self, word_list):
        return self.tokenizer.tokenize(' '.join(word_list))

    def infer(self, sample):
        model = self.model
        model.eval()
        item = model.sentence_encoder.tokenize(sample)
        item = (i.to(self.device) for i in item)
        logits = model.forward(*item)
        logits = model.softmax(logits)
        score, pred = logits.max(-1)
        return self.id2rel[pred.item()], score.item()

    def get_prob(self, input_):
        ret = []
        self.model.eval()
        correct_answer = np.zeros(len(self.id2rel))
        correct_answer[self.current_label] = 1.0
        for sent in input_:
            sent = sent.lower()
            valid = True
            for special in ['unused0', 'unused1', 'unused2', 'unused3']:
                if sent.count(special) != 1:
                    valid = False
                    break
            # Ignore sentences whose special tokens are not valid!
            if not valid:
                ret.append(correct_answer)
                continue
            # Convert data instance to sample
            sample = data2sample(DataInstance(x=sent, y=self.current_label),
                                 self.id2rel)
            # Predict sample label
            items = self.model.sentence_encoder.tokenize(sample)
            items = (i.to(self.device) for i in items)
            with torch.no_grad():
                logits = self.model.forward(*items)
                logits = self.model.softmax(logits).squeeze(0).cpu().numpy()
            ret.append(logits)

        return np.array(ret)


def sample2data(sample, rel2id):
    # Convert a single sample to a DataInstance
    # Process the sentence by adding indicating tokens to head / tail tokens
    tokens = sample['token']
    h_pos, t_pos = sample['h']['pos'], sample['t']['pos']
    head, tail = ' '.join(tokens[h_pos[0]:h_pos[1]]), ' '.join(
        tokens[t_pos[0]:t_pos[1]])
    rev = h_pos[0] > t_pos[0]
    if rev:
        # sent1, tail, sent2, head, sent3
        sent1 = ' '.join(tokens[:t_pos[0]])
        sent2 = ' '.join(tokens[t_pos[1]:h_pos[0]])
        sent3 = ' '.join(tokens[h_pos[1]:])
        sent = ' '.join((sent1, 'unused2', tail, 'unused3',
                         sent2, 'unused0', head, 'unused1', sent3))
    else:
        # sent1, head, sent2, head, sent3
        sent1 = ' '.join(tokens[:h_pos[0]])
        sent2 = ' '.join(tokens[h_pos[1]:t_pos[0]])
        sent3 = ' '.join(tokens[t_pos[1]:])
        sent = ' '.join((sent1, 'unused0', head, 'unused1',
                         sent2, 'unused2', tail, 'unused3', sent3))
    # Remove '_' chars
    sent = re.sub('_', '', sent)
    return DataInstance(x=sent,
                        y=rel2id[sample['relation']])


def data2sample(data, id2rel):
    sent, rel = data.x.lower(), id2rel[data.y]
    # Convert into a legal sample
    pos0, pos1, pos2, pos3 = [sent.find(w) for w in
                              ['unused0', 'unused1', 'unused2', 'unused3']]
    rev = pos0 > pos2
    h, t = sent[pos0 + len('unused0'):pos1], sent[pos2 + len('unused2'):pos3]
    if rev:
        s1, s2 = sent[:pos2], sent[pos3 + len('unused3'):pos0]
        s3 = sent[pos1 + len('unused1'):]
    else:
        s1, s2 = sent[:pos0], sent[pos1 + len('unused1'):pos2]
        s3 = sent[pos3 + len('unused3'):]
    # Convert string to token ids
    h, t, s1, s2, s3 = [part.strip().split()
                        for part in [h, t, s1, s2, s3]]
    if rev:
        words = s1 + t + s2 + h + s3
        h_pos = [len(s1) + len(t) + len(s2)]
        h_pos.append(h_pos[0] + len(h))
        t_pos = [len(s1)]
        t_pos.append(t_pos[0] + len(t))
    else:
        words = s1 + h + s2 + t + s3
        h_pos = [len(s1)]
        h_pos.append(h_pos[0] + len(h))
        t_pos = [len(s1) + len(h) + len(s2)]
        t_pos.append(t_pos[0] + len(t))
    return {'token': words, 'h': {'pos': h_pos}, 't': {'pos': t_pos}, 'relation': rel}


def sample2dataset(sample_list, rel2id):
    # Convert list of samples to dataset object
    data_list = []

    for sample in sample_list:
        data = sample2data(sample, rel2id)
        data_list.append(data)
    dataset = Dataset(data_list=data_list)

    return dataset


def dataset2sample(dataset, id2rel):
    # Convert dataset object to list of samples
    sample_list = []

    for data in dataset:
        sample = data2sample(data, id2rel)
        sample_list.append(sample)

    return sample_list


if __name__ == "__main__":
    # Test sample-data convert functions
    fin, fout = sys.argv[1:]
    rel2id = json.load(open('../data/tacred/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    samples = []
    for line in open(fin, 'r').readlines():
        samples.append(json.loads(line.strip()))
    with open(fout, 'w') as f:
        for sample in samples:
            f.write('original:' + json.dumps(sample) + '\n')
            data = sample2data(sample, rel2id)
            sample2 = data2sample(data, id2rel)
            f.write('after  2:' + json.dumps(sample2) + '\n')
            f.write('after  1:' + data.x + '\n')
            try:
                assert sample2['h']['pos'] == sample['h']['pos'] and sample2['t']['pos'] == sample['t']['pos']
            except Exception:
                print(sample)
                print(sample2)
                print()
