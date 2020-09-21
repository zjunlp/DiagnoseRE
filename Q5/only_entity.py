import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing original dataset is')
parser.add_argument('--output_file', '-o', type=str, default=None,
                    help='Where output moified dataset')

args = parser.parse_args()

# Modify dataset
logging.info('Loading dataset from ' + args.input_file)
dataset = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line.strip())
        h_pos, t_pos = sample['h']['pos'], sample['t']['pos']
        # Leave out contexts while keeping original entity order
        if h_pos[0] < t_pos[0]:
            token = sample['token'][h_pos[0]:h_pos[1]]
            token += sample['token'][t_pos[0]:t_pos[1]]
            new_h_pos = [0, h_pos[1] - h_pos[0]]
            new_t_pos = [new_h_pos[1], len(token)]
        else:
            token = sample['token'][t_pos[0]:t_pos[1]]
            token += sample['token'][h_pos[0]:h_pos[1]]
            new_t_pos = [0, t_pos[1] - t_pos[0]]
            new_h_pos = [new_t_pos[1], len(token)]
        dataset.append({'token': token,
                        'h': {'pos': new_h_pos},
                        't': {'pos': new_t_pos},
                        'relation': sample['relation']})

logging.info('Dumping modified dataset to ' + args.output_file)
with open(args.output_file, 'w') as f:
    for sample in dataset:
        f.write(json.dumps(sample))
        f.write('\n')
