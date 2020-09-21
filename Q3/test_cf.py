import torch
import json
import pickle
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Training for wiki80 and tacred dataset
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Full path for saving weights during training')
parser.add_argument('--test_path', '-t', type=str, required=True,
                    help='Full path to file containing testing data')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')
parser.add_argument('--batch_size', '-b', type=int, default=64,
                    help='Batch size for training and testing')

args = parser.parse_args()

# Some basic settings
root_path = '../'
sys.path.append(root_path)
rel2id = json.load(open(args.relation_path))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=args.max_seq_len,
    pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'),
    mask_entity=False
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model.to(torch.device('cuda:0'))

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.test_path,
    val_path=args.test_path,
    test_path=args.test_path,
    model=model,
    ckpt=args.model_path,
    batch_size=args.batch_size,
    max_epoch=1,
    lr=2e-5,
    opt='adamw'
)


def tokenize(item, **kwargs):
    tokens = item['bert_token']
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    # Find end of sentence: [SEP] token
    sent_len = len(tokens)
    if '[SEP]' in tokens:
        sent_len = tokens.index('[SEP]') + 1
    token_ids = torch.tensor(token_ids).long().unsqueeze(0)
    attn_mask = torch.zeros((1, args.max_seq_len), dtype=torch.long)
    attn_mask[0, :sent_len] = 1

    return token_ids, attn_mask


def eval_na(pred_result, use_name=False):
    # With `no_relation` label counted, the micro f1 result equals to precision
    correct = 0
    total = len(test_data)
    for i in range(total):
        if use_name:
            golden = test_data[i]['relation']
        else:
            golden = rel2id[test_data[i]['relation']]
        if golden == pred_result[i]:
            correct += 1
    acc = float(correct) / float(total)
    result = {'acc': acc, 'micro_p': acc,
              'micro_r': acc, 'micro_f1': acc}
    logging.info('Evaluation result: {}.'.format(result))

    # import pdb
    # pdb.set_trace()
    return result


# Modify the tokenization part so that the bert tokens are straightly encoded
bert_tokenizer = framework.model.sentence_encoder.tokenizer
framework.test_loader.dataset.tokenizer = tokenize

# Load test data
test_data = []
with open(args.test_path, 'r') as f:
    for line in f.readlines():
        test_data.append(json.loads(line.strip()))

# Modify the evaluation part so that the f1 score of `no_relation` is calculated
framework.test_loader.dataset.eval = eval_na

framework.load_state_dict(torch.load(args.model_path)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
