import numpy as np
import torch
from torch import nn
import json
import opennre
from opennre.model import SoftmaxNN
from tqdm import tqdm
import sys
import os
import argparse
import logging

device = torch.device('cuda:0')
logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Full path for loading weights of model to attack')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--output_file', '-o', type=str, default=None,
                    help='Where to store token information with gradient')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')

args = parser.parse_args()

# Use hook to fetch gradient of intermediate variables
embed = {}


def get_embed(name):
    def hook(model, input, output):
        embed[name] = output
    return hook


# Some basic settings
root_path = '../'
sys.path.append(root_path)
device = torch.device('cuda:0')
rel2id = json.load(open(args.relation_path, 'r'))
id2rel = {v: k for k, v in rel2id.items()}

# Load dataset
samples = []
with open(args.input_file, 'r') as f:
    for line in tqdm(f.readlines(), desc='reading dataset'):
        sample = json.loads(line)
        samples.append(sample)

# Make and load model
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=args.max_seq_len,
    pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'),
    mask_entity=False
)
model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model.load_state_dict(torch.load(args.model_path)['state_dict'])
model.to(device)

# Get parts of sentence encoder model
bert_model = model.sentence_encoder.bert
tokenizer = model.sentence_encoder.tokenizer
sentence_encoder.bert.embeddings.register_forward_hook(get_embed('embeddings'))
word_emb = bert_model.get_input_embeddings().weight
ce = torch.nn.CrossEntropyLoss()
model.eval()

# Main part of operations
for sample in tqdm(samples, desc='Processing test sentences...'):
    # Change value of model embedding
    alpha_levels = 8
    all_grad = []
    # Get sentence embedding
    inputs = model.sentence_encoder.tokenize(sample)
    inputs = [i.to(device) for i in inputs]
    token_ids = inputs[0][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    for alpha in range(alpha_levels + 1):
        # Reinitializing embeddings
        alpha_rate = 1.0 * alpha / alpha_levels
        bert_model.set_input_embeddings(
            nn.Embedding.from_pretrained(word_emb * alpha_rate))

        # Get predict score
        # Modified codes in modeling_bert.py so that the embedding is returned
        outputs = bert_model(*inputs)
        token_embed = embed['embeddings']

        # Important step: keep gradient for intermediate variables
        token_embed.retain_grad()
        _, logits = outputs
        logits = model.softmax(model.fc(model.drop(logits)))

        # Record model prediction
        if alpha == alpha_levels:
            _, pred = logits.max(-1)
            label = id2rel[int(pred[0].cpu())]
            correct = label == sample['relation']

        # Backward to get gradient
        loss = ce(logits, torch.LongTensor(
            [rel2id[sample['relation']]]).to(device))
        loss.backward()
        all_grad.append(token_embed.grad)

    grad_mean = torch.mean(torch.cat(all_grad, dim=0),
                           dim=0).squeeze(0).cpu().numpy()

    # Calculate norm of hidden dims
    grad = np.sqrt(grad_mean ** 2).sum(axis=1)
    grad = (grad - grad.min()) / (grad.max() - grad.min())

    # Merge tokens with gradients (aligned to tokens' length)
    result = list(zip(tokens, grad[:min(len(tokens), args.max_seq_len)]))
    with open(args.output_file, 'a') as f:
        f.write(json.dumps({'token_grad': [[t, str(g)] for t, g in result],
                            'correct': correct}))
        f.write('\n')
