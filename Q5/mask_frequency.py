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

args = parser.parse_args()

# Modify dataset
dataset = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        dataset.append(sample)
logging.info('Loaded dataset ({} samples) from {}'.format(
    len(dataset), args.input_file))

# Calculate entity pair's frequency
ent_pairs = {}
for idx, sample in enumerate(dataset):
    h_pos, t_pos = sample['h']['pos'], sample['t']['pos']
    h = ' '.join(sample['token'][h_pos[0]:h_pos[1]]).lower()
    t = ' '.join(sample['token'][t_pos[0]:t_pos[1]]).lower()
    pair = h, t
    if pair not in ent_pairs:
        ent_pairs[pair] = []
    ent_pairs[pair].append(idx)

ent_freq = {pair: len(indices) for pair, indices in ent_pairs.items()}
max_freq = max([freq for freq in ent_freq.values()])
min_freq = min([freq for freq in ent_freq.values()])
mask_rate = {pair: (freq - min_freq) / (max_freq - min_freq)
             for pair, freq in ent_freq.items()}

# Mask entities by frequency
total_mask = 0
for pair, rate in mask_rate.items():
    indices = ent_pairs[pair]
    # At least leave one sample per pair of entity
    mask_count = min(len(indices) - 1, max(int(len(indices) * rate), 1))
    random.shuffle(indices)
    for idx in indices[:mask_count]:
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
    total_mask += mask_count
logging.info('Masked {} samples out of {} pairs'.format(
    total_mask, len(ent_pairs)))

logging.info('Dumping modified dataset ({} samples) to {}'.format(
    len(dataset), args.output_file))
with open(args.output_file, 'w') as f:
    for sample in dataset:
        f.write(json.dumps(sample))
        f.write('\n')
