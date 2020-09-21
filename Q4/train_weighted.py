import torch
import json
import pickle
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging
import math
from torch.utils.data import WeightedRandomSampler, DataLoader

logging.basicConfig(level=logging.INFO)

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Training for wiki80 and tacred dataset
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Full path for saving weights during training')
parser.add_argument('--restore', action='store_true',
                    help='Whether to restore model weights from given model path')
parser.add_argument('--train_path', '-t', type=str, required=True,
                    help='Full path to file containing training data')
parser.add_argument('--valid_path', '-v', type=str, required=True,
                    help='Full path to file containing validation data')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--rule_path', '--rule', type=str, required=True,
                    help='Path to file containing rules for generating weights of training samples')
parser.add_argument('--num_epochs', '-e', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')
parser.add_argument('--batch_size', '-b', type=int, default=64,
                    help='Batch size for training and testing')
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric chosen for evaluation')

args = parser.parse_args()

# Some basic settings
root_path = '../'
sys.path.append(root_path)
rel2id = json.load(open(args.relation_path))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=args.max_seq_len,
    pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'),
    mask_entity=False
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_path,
    val_path=args.valid_path,
    test_path=args.valid_path,
    model=model,
    ckpt=args.model_path,
    batch_size=args.batch_size,
    max_epoch=args.num_epochs,
    lr=2e-5,
    opt='adamw'
)

# Modify the `train_loader`'s sampler with a weighted random sampler
logging.info(
    'Calculate and assign weights for training sample by given rules...')

# Load rules
rules = json.load(open(args.rule_path, 'r'))
for rel in rules:
    rules[rel]['rules'] = set(rules[rel]['rules'])
train_samples = []

# Load training data
with open(args.train_path, 'r') as f:
    for line in f.readlines():
        train_samples.append(json.loads(line.strip()))

# Calculate samples' weights
sample_weights = []
for sample in train_samples:
    rel = sample['relation']
    token_set = set([token.lower() for token in sample['token']])
    if rel in rules:
        common_tokens = token_set.intersection(rules[rel]['rules'])
    # With one token appeared in rules, reduce half probability of choosing this sample
    sample_weights.append(math.pow(0.5, len(common_tokens)))

train_sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(train_samples))

# Replace original training sampler
train_dataset = opennre.framework.SentenceREDataset(
    path=args.train_path, rel2id=rel2id, tokenizer=sentence_encoder.tokenize, kwargs={})
framework.train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=8,
                                    collate_fn=opennre.framework.SentenceREDataset.collate_fn)

# Restore from old model weights
if args.restore:
    framework.load_state_dict(torch.load(args.model_path)['state_dict'])
    logging.info('Restored model weights from {}'.format(args.model_path))

# Train the model
framework.train_model(args.metric)
