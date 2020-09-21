import os
import json

if not os.path.exists('normal'):
    os.mkdir('normal')

for data_type in ['train', 'dev', 'test']:
    with open('{}.json'.format(data_type), 'r') as f:
        data = json.load(f)
    dataset = []
    for sample in data:
        token = sample['token']
        h_pos = [sample['subj_start'], sample['subj_end'] + 1]
        t_pos = [sample['obj_start'], sample['obj_end'] + 1]
        relation = sample['relation']
        dataset.append({'token': token,
                        'h': {'pos': h_pos},
                        't': {'pos': t_pos},
                        'relation': relation})
    with open('normal/{}.txt'.format(data_type), 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample))
            f.write('\n')
