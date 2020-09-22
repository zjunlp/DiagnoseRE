import json
import argparse
import logging
import math
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing dataset is')
parser.add_argument('--rule_file', '-r', type=str, required=True,
                    help='Where the json file containing substitution rule_dict is')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Where to store token information with gradient')
args = parser.parse_args()

# Load rule file
rule_dict = json.load(open(args.rule_file, 'r'))
for rel in rule_dict:
    rule_dict[rel]['rules'] = set(rule_dict[rel]['rules'])
logging.info('Loaded {} rules from {}'.format(len(rule_dict), args.input_file))

# Load samples
samples = []
sub_count = 0
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        # Perform substitution
        if sample['relation'] in rule_dict:
            rule_set = rule_dict[sample['relation']]['rules']
            sub_indices = []
            for idx, token in enumerate(sample['token']):
                token = token.lower()
                if token in rule_set:
                    sub_indices.append(idx)
            sub_count += len(sub_indices) > 0
            if sub_indices:
                for idx in sub_indices:
                    sample['token'][idx] = ''
        samples.append(sample)

# Write samples
with open(args.output_file, 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')
logging.info('Masked {}/{} samples'.format(sub_count, len(samples)))
