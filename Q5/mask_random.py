import json
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--output_file', '-o', type=str, default=None,
                    help='Where output moified dataset')
parser.add_argument('--keep_origin', '-k', action='store_true',
                    help='Whether to keep original samples in output dataset')
parser.add_argument('--ratio', '-r', type=float, default=1.0,
                    help='Ratio of random masking')

args = parser.parse_args()

# Load dataset
dataset = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        dataset.append(sample)
logging.info('Loaded dataset ({} samples) from {}'.format(
    len(dataset), args.input_file))

# Mask entities randomly with ratio
indices = list(range(0, len(dataset)))
random.shuffle(indices)
if args.ratio < 1:
    indices = indices[:int(len(dataset) * args.ratio)]

for idx in indices:
    sample = dataset[idx]
    h_pos, t_pos = sample['h']['pos'], sample['t']['pos']
    # Just remove the entities' words while keeping original entity order
    if h_pos[0] < t_pos[0]:
        token = sample['token'][:h_pos[0]]
        new_h_pos = [h_pos[0]] * 2
        token += sample['token'][h_pos[1]:t_pos[0]]
        new_t_pos = [len(token)] * 2
        token += sample['token'][t_pos[1]:]
    else:
        token = sample['token'][:t_pos[0]]
        new_t_pos = [t_pos[0]] * 2
        token += sample['token'][t_pos[1]:h_pos[0]]
        new_h_pos = [len(token)] * 2
        token += sample['token'][h_pos[1]:]
    # Replace original sample
    new_sample = {'token': token,
                  'h': {'pos': new_h_pos},
                  't': {'pos': new_t_pos},
                  'relation': sample['relation']}
    if args.keep_origin:
        dataset.append(new_sample)
    else:
        dataset[idx] = new_sample

logging.info('Dumping modified dataset ({} samples) to {}'.format(
    len(dataset), args.output_file))
with open(args.output_file, 'w') as f:
    for sample in dataset:
        f.write(json.dumps(sample))
        f.write('\n')
