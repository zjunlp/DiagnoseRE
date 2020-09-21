import os
import sys
import json
import argparse
import logging
import random
import torch
import opennre
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--original_data', '--origin', dest='origin', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Where to store token information with gradient')
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Where the model for wrong prediction is')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')

args = parser.parse_args()

# Load dataset
dataset = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        dataset.append(sample)
origin_dataset = []
with open(args.origin, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        origin_dataset.append(sample)

assert len(dataset) == len(origin_dataset)
logging.info('Loaded dataset ({} samples) from {}'.format(
    len(dataset), args.input_file))
rel2id = json.load(open(args.relation_path, 'r'))
id2rel = {v: k for k, v in rel2id.items()}

# Load model
root_path = '../'
sys.path.append(root_path)
device = torch.device('cuda:0')

# Load original BERT model
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=args.max_seq_len,
    pretrain_path=os.path.join(
        root_path, 'pretrain/bert-base-uncased'),
    mask_entity=False)
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model.to(device)
model.load_state_dict(torch.load(args.model_path)['state_dict'])
model.eval()

# Inference samples and select wrong samples
wrong_samples = []
idx = 0
for sample in tqdm(dataset, desc='Testing samples...'):
    item = model.sentence_encoder.tokenize(sample)
    item = (i.to(device) for i in item)
    logits = model.forward(*item)
    logits = model.softmax(logits)
    score, pred = logits.max(-1)
    pred_label = id2rel[pred.item()]
    if pred_label != sample['relation']:
        # Add original data
        wrong_samples.append(origin_dataset[idx])
    idx += 1

logging.info('Dumping wrongly predicted {} samples to {}'.format(
    len(wrong_samples), args.output_file))
with open(args.output_file, 'w') as f:
    for sample in wrong_samples:
        f.write(json.dumps(sample))
        f.write('\n')
