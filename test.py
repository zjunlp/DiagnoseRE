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
root_path = '.'
sys.path.append(root_path)
opennre.download('bert_base_uncased', root_path=root_path)
model_path = args.model_path
os.makedirs(model_path[:model_path.rfind('/')], exist_ok=True)
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

framework.load_state_dict(torch.load(args.model_path)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
