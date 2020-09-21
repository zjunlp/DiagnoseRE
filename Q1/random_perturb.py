import os
import argparse
import json
import nltk
from collections import deque
from tqdm import tqdm
import random
import logging
import checklist
import torch
import spacy
from checklist.perturb import Perturb
# from checklist.editor import Editor

spacy_model_path = 'en_core_web_sm-2.3.1/en_core_web_sm/en_core_web_sm-2.3.1'
# editor = Editor()
nlp = spacy.load(spacy_model_path)
logging.basicConfig(level=logging.INFO)


def read_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in tqdm(f.readlines(), desc='reading dataset'):
            dataset.append(json.loads(line))
    return dataset


# def perturb_context_with_synonyms(example):
#     # This is deprecated because the synonyms API is buggy...
#     replaced_samples = []
#     tokens = example['token']
#     relation = example['relation']
#     head_pos, tail_pos = example['h']['pos'], example['t']['pos']
#     # Find position of context between two entities
#     if head_pos[0] < tail_pos[0]:
#         start, end = head_pos[1], tail_pos[0]
#     else:
#         start, end = tail_pos[1], head_pos[0]
#     words_to_replace = tokens[start:end]
#     left, right = tokens[:start], tokens[end:]

#     # Generate synonyms for context words
#     example_sentence = ' '.join(tokens)
#     for idx, word in enumerate(words_to_replace):
#         synonym_list = editor.synonyms(example_sentence, word)
#         for synonym in synonym_list:
#             perturb_tokens = left + words_to_replace[:idx] + [synonym]
#             perturb_tokens += words_to_replace[idx + 1:] + right
#             perturb_sample = {'token': perturb_tokens,
#                               'relation': relation,
#                               'h': {'pos': head_pos},
#                               't': {'pos': tail_pos}}
#             replaced_samples.append(perturb_sample)
#     return replaced_samples


def perturb_context_by_type(example, locations):
    def merge(sent_dict):
        # Merge sentence parts in order
        sent_order = ['sent1', 'ent1', 'sent2', 'ent2', 'sent3']
        q = deque([[sent] for sent in sent_dict[sent_order[0]]])
        curr_idx = 1
        while curr_idx < len(sent_order):
            curr_len = len(q)
            for _ in range(curr_len):
                prev_sents = q.pop()
                for postfix in sent_dict[sent_order[curr_idx]]:
                    q.appendleft(prev_sents + [postfix])
            curr_idx += 1
        return list(q)

    # Perturb context by different types:
    # Punctuation, Typos, Contractions, Changing names,
    # Change locations, Change numbers
    replaced_samples = []
    tokens = example['token']
    relation = example['relation']
    head_pos, tail_pos = example['h']['pos'], example['t']['pos']
    rev = head_pos[0] > tail_pos[0]
    # Split the tokens
    sent1, ent1, sent2, ent2, sent3 = (' '.join(tokens[:head_pos[0]]),
                                       ' '.join(
                                           tokens[head_pos[0]:head_pos[1]]),
                                       ' '.join(
                                           tokens[head_pos[1]:tail_pos[0]]),
                                       ' '.join(
                                           tokens[tail_pos[0]:tail_pos[1]]),
                                       ' '.join(tokens[tail_pos[1]:]))
    if rev:
        # Reversed order: tail appears before head
        ent1, ent2 = ent2, ent1
    # Pack all parts into a dict and modify by names
    sent_dict = {'sent1': [sent1], 'ent1': [ent1], 'sent2': [sent2],
                 'ent2': [ent2], 'sent3': [sent3]}
    for perturb_func in [Perturb.punctuation, Perturb.strip_punctuation,
                         Perturb.change_location, Perturb.change_names, Perturb.change_number,
                         Perturb.contractions, Perturb.add_typos]:
        sent_dict_copy = sent_dict.copy()
        # Diverge
        for loc in locations:
            origin = sent_dict[loc][0]
            if not origin:
                # No tokens given
                continue
            if perturb_func not in {Perturb.contractions, Perturb.add_typos}:
                # Process string to spacy.Doc
                origin = nlp(origin)
            if perturb_func in {Perturb.strip_punctuation, Perturb.punctuation}:
                # All tokens are useless
                if [tok.pos_ for tok in origin] == ['PUNCT'] * len(origin):
                    continue
            if perturb_func == Perturb.add_typos and len(origin) == 1:
                # At least 2 tokens are needed
                continue
            if perturb_func in {Perturb.change_location, Perturb.change_number, Perturb.change_names}:
                # Control the number of perturbed samples
                ret = perturb_func(origin, n=1)
            else:
                ret = perturb_func(origin)

            # Process result
            if not ret:
                # Returned nothing
                ret = [sent_dict[loc][0]]
            if isinstance(ret, str):
                # Wrap single sentence
                ret = [ret]
            sent_dict_copy[loc] = ret

        # Merge all parts of perturbed sentences and filter out original sentence
        for merged_sent in filter(lambda perturbed_tokens: perturbed_tokens != tokens,
                                  merge(sent_dict_copy)):
            tokens = ' '.join(merged_sent).split(' ')
            sent1, ent1, sent2, ent2, sent3 = merged_sent
            head_pos = [len(sent1.split(' '))]
            head_pos.append(head_pos[0] + len(ent1.split(' ')))
            tail_pos = [head_pos[1] + len(sent2.split(' '))]
            tail_pos.append(tail_pos[0] + len(ent2.split(' ')))
            if rev:
                head_pos, tail_pos = tail_pos, head_pos
            replaced_samples.append({
                'relation': relation,
                'token': tokens,
                'h': {'pos': head_pos},
                't': {'pos': tail_pos},
                'perturb': perturb_func.__name__
            })
    return replaced_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str,
                        help='Input file containing dataset')
    parser.add_argument('--output_file', '-o', type=str,
                        help='Output file containing perturbed dataset')
    parser.add_argument('--locations', '-l', nargs='+',
                        choices=['sent1', 'sent2', 'sent3', 'ent1', 'ent2'],
                        help='List of positions that you want to perturb')

    args = parser.parse_args()
    logging.info(str(args))

    # Load and perturb
    origin_data = read_dataset(args.input_file)
    with open(args.output_file, 'w') as f:
        for sample in tqdm(origin_data, desc='perturbing dataset'):
            ret = Perturb.perturb([sample], perturb_context_by_type,
                                  keep_original=False, locations=args.locations)
            # Write original sample
            f.write(json.dumps(sample))
            f.write('\n')
            for sample_list in ret.data:
                for sample in sample_list:
                    f.write(json.dumps(sample))
                    f.write('\n')
