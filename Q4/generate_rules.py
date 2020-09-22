import json
import argparse
import logging
import math
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', nargs='+', type=str, required=True,
                    help='Where the input file containing dataset is')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Where to store token information with gradient')
parser.add_argument('--num_rules', '-n', type=int, default=5,
                    help='Maximum number of key words chosen as rules for each relation')

args = parser.parse_args()

# Load datasets
relation_tokens = {}
sentence_count = {}
for fname in args.input_file:
    with open(fname, 'r') as f:
        for line in f.readlines():
            sample = json.loads(line)
            rel = sample['relation']
            if rel not in relation_tokens:
                relation_tokens[rel] = {}
                sentence_count[rel] = 0
            sentence_count[rel] += 1
            for token in sample['token']:
                token = token.lower()
                if token not in relation_tokens[rel]:
                    relation_tokens[rel][token] = 0
                relation_tokens[rel][token] += 1

# Calculate tf-idf values
token_idf = {}
tfidf = {rel: {} for rel in relation_tokens.keys()}
for rel, token_count in tqdm(relation_tokens.items(), desc='Calculating tf-idf values...'):
    # total = sum(token_count.values())
    for token in token_count.keys():
        # Calculate tf: using modified tf, frequency per sentence
        tf = token_count[token] / sentence_count[rel]
        # Calculate idf
        if token not in token_idf:
            doc_count = sum([int(token in doc)
                             for doc in relation_tokens.values()])
            idf = math.log(len(relation_tokens) / (1 + doc_count))
            token_idf[token] = idf
        else:
            idf = token_idf[token]
        # Calculate score
        tfidf[rel][token] = tf * idf

# Generate rules by tf-idf values of tokens in each relation
with open(args.output_file, 'w') as f:
    rules = {}
    for rel, token_value in tfidf.items():
        token_pairs = [(token, count) for token, count in token_value.items()]
        token_pairs = sorted(token_pairs, key=lambda pair: -pair[1])
        # The rules could be manually check fine-grainedly
        rules[rel] = {
            'sentences': sentence_count[rel],
            'rules': [token for token, score in token_pairs[:args.num_rules]]
        }
    f.write('{')
    rel_count = 0
    for rel in sorted(rules.keys(), key=lambda rel: -rules[rel]['sentences']):
        f.write('"{}":{}'.format(rel, json.dumps(rules[rel])))
        rel_count += 1
        if rel_count < len(rules):
            f.write(',')
        f.write('\n')
    f.write('}')
