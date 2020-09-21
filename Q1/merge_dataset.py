import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', '-i', nargs='+',
                    help='List of input files containing datasets to merge')
parser.add_argument('--output_file', '-o',
                    help='Output file containing merged dataset')

args = parser.parse_args()

reduced_samples = 0
all_samples = set()
for fname in args.input_files:
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line in all_samples:
                reduced_samples += 1
            else:
                all_samples.add(line)
print('Reduced {} repeated samples'.format(reduced_samples))
with open(args.output_file, 'w') as f:
    for line in tqdm(all_samples):
        f.write(line)
