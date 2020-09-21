import time
from multiprocessing import Queue, Process, set_start_method
from victim import *

# Enable this to allow tensors being converted to numpy arrays
tf.compat.v1.enable_eager_execution()
try:
    set_start_method('spawn')
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def attack_process(idx, args, q):
    rel2id = json.load(open(args.relation_path, 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    total_count, success_count, total_time = 0, 0, 0.0

    # Load victim model
    # Distribute models on devices equally
    device_id = idx % torch.cuda.device_count()  # Start from 0
    device = torch.device('cuda:' + str(device_id))
    model = REClassifier(
        args.max_seq_len, args.model_path, rel2id, id2rel, device)
    logging.info('Build model ' + str(idx) + ' on device ' + str(device_id))

    # Load attacker
    logging.info('New Attacker ' + str(idx))

    # Preserve special tokens
    skip_words = ['unused0', 'unused1', 'unused2', 'unused3']

    # Attacker models
    attack_models = {
        'pw': OpenAttack.attackers.PWWSAttacker,
        'tf': OpenAttack.attackers.TextFoolerAttacker,
        'hf': OpenAttack.attackers.HotFlipAttacker,
        'uat': OpenAttack.attackers.UATAttacker
    }
    if args.attacker != 'uat':
        attacker = attack_models[args.attacker](skip_words=skip_words)
    else:
        attacker = attack_models[args.attacker]()

    # Build evaluation object
    options = {"success_rate": False, "fluency": False, "mistake": False, "semantic": False, "levenstein": False,
               "word_distance": False, "modification_rate": False, "running_time": False, "progress_bar": False,
               "invoke_limit": 500, "average_invoke": True}
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(
        attacker, model, **options)

    # Generate samples in batches
    while True:
        if q.empty():
            break
        data = q.get()
        # Save label for current sample for reference
        model.current_label = data.y
        start_time = time.time()
        adv_data = attack_eval.generate_adv([data])
        sample_list = dataset2sample(adv_data, id2rel)
        with open(args.output_file, 'a') as f:
            for sample in sample_list:
                f.write(json.dumps(sample) + '\n')
        cost_time = time.time() - start_time
        total_time += cost_time
        total_count += 1
        success_count += len(sample_list)
        logging.info('Success:{}/{:02.2f}%, time:{:02.2f}s/{:02.2f}s, jobs:{}/{}'.format(
            len(sample_list), success_count / total_count * 100,
            cost_time, total_time / total_count,
            total_count, q.qsize()))

    logging.info('Attacker {} finished and quit.'.format(idx))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='Where the input file containing original dataset is')
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help='Full path for loading weights of model to attack')
    parser.add_argument('--relation_path', '-r', type=str, required=True,
                        help='Full path to json file containing relation to index dict')
    parser.add_argument('--attacker', '-a', type=str, choices=['pw', 'tf', 'hf', 'uat'], default='pw',
                        help='Name of attacker model, pw = PWWS, tf = TextFooler, hf = HotFlip')
    parser.add_argument('--output_file', '-o', type=str, required=True,
                        help='Where to store adverserial dataset is')
    parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                        help='Maximum sequence length of bert model')
    parser.add_argument('--num_jobs', '-j', type=int, default=1,
                        help='Maximum number of parallel workers in attacking')
    parser.add_argument('--start_index', '-s', type=int, default=0,
                        help='Index of sample to start processing, used when you want to restore progress')
    args = parser.parse_args()

    logging.info('CUDA device status: {}, devices: {}'.format(
        torch.cuda.is_available(), torch.cuda.device_count()))

    # Load dataset
    logging.info('Load dataset')
    samples = []
    rel2id = json.load(open(args.relation_path, 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    with open(args.input_file, 'r') as f:
        for line in tqdm(f.readlines(), desc='reading dataset'):
            sample = json.loads(line)
            samples.append(sample)
    dataset = sample2dataset(samples, rel2id)

    # Cut dataset into mini-batches, each containing fixed number of samples
    logging.info('Creating queue for dataset...')
    queue = Queue()
    for start_idx in range(args.start_index, len(dataset)):
        queue.put(dataset[start_idx])
    logging.info('Total tasks: ' + str(queue.qsize()))

    # Start attacking
    logging.info('Start attack')
    if args.num_jobs > 1:
        # Multi-process attacking
        process_list = []
        for index in range(args.num_jobs):
            p = Process(target=attack_process, args=(index + 1, args, queue))
            process_list.append(p)
            p.start()
        for p in process_list:
            p.join()
    else:
        # Single-process attacking
        attack_process(0, args, queue)
