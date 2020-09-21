import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='File containing tokenized dataset')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='File to output fixed dataset')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Number of tokens in a sample')

args = parser.parse_args()

fix_count = 0
# Load dataset
dataset = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        dataset.append(json.loads(line.strip()))

# Fix token lengths
for sample in dataset:
    token = sample['bert_token']
    if len(token) < args.max_seq_len:
        token += ['[PAD]'] * (args.max_seq_len - len(token))
        fix_count += 1
    elif len(token) > args.max_seq_len:
        token = token[:args.max_seq_len]
        fix_count += 1
    assert len(token) == args.max_seq_len
    sample['bert_token'] = token

# Dump dataset
with open(args.output_file, 'w') as f:
    for sample in dataset:
        f.write(json.dumps(sample) + '\n')
logging.info('Fixed length of {}/{} samples to {}'.format(
    fix_count, len(dataset), args.max_seq_len))
