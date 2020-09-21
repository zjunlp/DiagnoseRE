import numpy as np
import json
from tqdm import tqdm
import sys
import torch
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--grad_file', '-g', type=str, required=True,
                    help='Where the input file containing tokens with gradient is')
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Where to store token information with gradient')
parser.add_argument('--keep_origin', '-k', action='store_true',
                    help='Whether to keep original samples(as tokenized samples)')
parser.add_argument('--num_tokens', '-n', type=int, default=1,
                    help='Maximum number of tokens masked(removed) in counter-factual samples')
parser.add_argument('--threshold', '-t', type=float, default=0.2,
                    help='Lower bound of key words, i.e. the minimum required weight of being a key word')

args = parser.parse_args()

# Load stop words
stop_words = set()
with open('stopwords.txt', 'r') as f:
    for line in f.readlines():
        stop_words.add(line.strip())
# Stop tokens: omitted for importance ranking
stop_token_heads = {'[', ',', '.', ';', '_', '|', '(', ')'
                    ':', '\'', '"', '!', '?', '-', '`', '#'}
neg = 'no_relation'

samples = []
with open(args.input_file, 'r') as f:
    for line in tqdm(f.readlines(), desc='reading dataset'):
        sample = json.loads(line)
        samples.append(sample)

token_samples, masked_token_samples = [], []
with open(args.grad_file, 'r') as f:
    for idx, line in tqdm(enumerate(f.readlines()), desc='Masking tokenized samples...'):
        token_sample = json.loads(line.strip())
        token_list = token_sample['token_grad']
        correct = token_sample['correct']
        tokens = [token for token, _ in token_list]
        sample = {'bert_token': tokens, 'relation': samples[idx]['relation']}
        token_samples.append(sample)

        # Only choose correct samples with relation to perform mask
        if sample['relation'] == neg or not correct:
            continue

        # Choose tokens with higher weights to mask
        ordered_tokens = sorted([(token, float(score), idx)
                                 for idx, (token, score) in enumerate(token_list)],
                                key=lambda triplet: -triplet[1])

        # Filter out stop words or special tokens
        filtered_tokens = list(filter(
            lambda triplet: triplet[0][0] not in stop_token_heads and triplet[0] not in stop_words,
            ordered_tokens))

        # Masked tokens' weights should be over given threshold
        filtered_tokens = list(filter(
            lambda triplet: triplet[1] >= args.threshold, filtered_tokens))

        if len(filtered_tokens) > 0:
            # Do not mask too many tokens
            filtered_tokens = filtered_tokens[:args.num_tokens]
            masked_token_list = token_list.copy()
            for triplet in filtered_tokens:
                masked_token_list[triplet[2]] = ()
            masked_tokens = [token for token,
                             _ in filter(len, masked_token_list)]
            # Pad to original length
            masked_tokens += ['[PAD]'] * len(filtered_tokens)
            masked_sample = {'bert_token': masked_tokens,
                             'relation': neg if args.num_tokens > 0 else sample['relation']}
            masked_token_samples.append(masked_sample)

logging.info('Generated {} masked samples, masked {} tokens each.'.format(
    len(masked_token_samples), args.num_tokens))
# logging.info('acc: ' + str(correct / len(samples)))
with open(args.output_file, 'w') as f:
    if args.keep_origin:
        for sample in token_samples:
            f.write(json.dumps(sample))
            f.write('\n')
    for sample in masked_token_samples:
        f.write(json.dumps(sample))
        f.write('\n')
