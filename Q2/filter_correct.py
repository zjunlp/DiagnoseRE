from victim import *

logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing full dataset is')
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Full path for loading weights of model to attack')
parser.add_argument('--relation_path', '-r', type=str, required=True,
                    help='Full path to json file containing relation to index dict')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Place to save samples that model predicts correctly')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')

args = parser.parse_args()

# Initialize model
rel2id = json.load(open(args.relation_path, 'r'))
id2rel = {v: k for k, v in rel2id.items()}
device = torch.device('cuda:0')
model = REClassifier(args.max_seq_len, args.model_path, rel2id, id2rel, device)

# Filter out correct samples
correct_samples = []
with open(args.input_file, 'r') as f:
    for line in tqdm(f.readlines(), desc='Filtering correct samples'):
        sample = json.loads(line)
        label, _ = model.infer(sample)
        if label != sample['relation']:
            continue
        correct_samples.append(sample)

# Dump correct samples
with open(args.output_file, 'w') as f:
    for sample in tqdm(correct_samples, desc='Dumping correct samples'):
        f.write(json.dumps(sample))
        f.write('\n')
logging.info('Dumped {} correct samples to {}'.format(
    len(correct_samples), args.output_file))
