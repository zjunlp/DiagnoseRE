# DiagnoseRE

This repository is the official implementation of the CCKS2021 paper [On Robustness and Bias Analysis of BERT-based Relation Extraction](https://arxiv.org/pdf/2009.06206.pdf).


## Requirements

- To install basic requirements:

```python
pip install requirements.txt
```

- Also `opennre==0.1` is required to run the codes and it could be installed by following the instructions [here](https://github.com/thunlp/OpenNRE).

## Datasets

- TACRED can be found here: https://nlp.stanford.edu/projects/tacred/
  - The raw tacred data is in `json` format and can be transformed using `data/tacred/convert_json.py`.

- Wiki80 can be downloaded using scripts here: https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_wiki80.sh

## Basic training & testing

To train a model with given dataset, run `train.py`:

```bash
$ python train.py -h
usage: train.py [-h] --model_path MODEL_PATH [--restore] --train_path
                TRAIN_PATH --valid_path VALID_PATH --relation_path
                RELATION_PATH [--num_epochs NUM_EPOCHS]
                [--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE]
                [--metric {micro_f1,acc}]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for saving weights during training
  --restore             Whether to restore model weights from given model path
  --train_path TRAIN_PATH, -t TRAIN_PATH
                        Full path to file containing training data
  --valid_path VALID_PATH, -v VALID_PATH
                        Full path to file containing validation data
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --num_epochs NUM_EPOCHS, -e NUM_EPOCHS
                        Number of training epochs
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for training and testing
  --metric {micro_f1,acc}
                        Metric chosen for evaluation
```

To test a model with a given dataset, run `test.py`:

```bash
$ python test.py -h
usage: test.py [-h] --model_path MODEL_PATH --test_path TEST_PATH
               --relation_path RELATION_PATH [--max_seq_len MAX_SEQ_LEN]
               [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for saving weights during training
  --test_path TEST_PATH, -t TEST_PATH
                        Full path to file containing testing data
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for training and testing
```

## Q1 Random Perturbation

This part is based on [checklist](https://github.com/marcotcr/checklist), and you may run the codes under `Q1` folder.

Also the `spaCy` model `en_core_web_sm-2.3.1` should be placed under `Q1` folder. The model can be downloaded by running: `python -m spacy download en_core_web_sm` .

Remember to modify the `spacy_model_path` in `random_pertueb.py` is necessary.

- To random perturb a dataset, run `Q1/random_perturb.py`:

```bash
$ python random_perturb.py -h
usage: random_perturb.py [-h] [--input_file INPUT_FILE]
                         [--output_file OUTPUT_FILE]
                         [--locations {sent1,sent2,sent3,ent1,ent2} [{sent1,sent2,sent3,ent1,ent2} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Input file containing dataset
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Output file containing perturbed dataset
  --locations {sent1,sent2,sent3,ent1,ent2} [{sent1,sent2,sent3,ent1,ent2} ...], -l {sent1,sent2,sent3,ent1,ent2} [{sent1,sent2,sent3,ent1,ent2} ...]
                        List of positions that you want to perturb
```

- To merge datasets and filter out common samples (you may want to do this when merging two perturbed datasets both keeping original samples in them), run `merge_dataset.py`:

```bash
$ python merge_dataset.py -h
usage: merge_dataset.py [-h] [--input_files INPUT_FILES [INPUT_FILES ...]]
                        [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --input_files INPUT_FILES [INPUT_FILES ...], -i INPUT_FILES [INPUT_FILES ...]
                        List of input files containing datasets to merge
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Output file containing merged dataset
```

- After data generation, you can train / test the model using `train.py` / `test.py`.

## Q2 Adversarial Perturbations

This part is based on [OpenAttack](https://github.com/thunlp/OpenAttack).

Adverserial sample generating on RE supports all the models in OpenAttack library like `PWWS,TextFooler,HotFlip` and so on. Also, the codes supports multi-process working to boost the sample generating process.

- You may filter samples that model predicts correctly before generating adverserial samples with them, by running `Q2/filter_correct.py`:

```bash
$ python filter_correct.py -h
usage: filter_correct.py [-h] --input_file INPUT_FILE --model_path MODEL_PATH
                         --relation_path RELATION_PATH --output_file
                         OUTPUT_FILE [--max_seq_len MAX_SEQ_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing full dataset is
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for loading weights of model to attack
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Place to save samples that model predicts correctly
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
```

- Using samples model can correctly predict, you can generate adverserial samples using `Q2/attack.py`:

```bash
$ python attack.py -h
usage: attack.py [-h] --input_file INPUT_FILE --model_path MODEL_PATH
                 --relation_path RELATION_PATH [--attacker {pw,tf,hf,uat}]
                 --output_file OUTPUT_FILE [--max_seq_len MAX_SEQ_LEN]
                 [--num_jobs NUM_JOBS] [--start_index START_INDEX]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for loading weights of model to attack
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --attacker {pw,tf,hf,uat}, -a {pw,tf,hf,uat}
                        Name of attacker model, pw = PWWS, tf = TextFooler, hf
                        = HotFlip
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store adverserial dataset is
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
  --num_jobs NUM_JOBS, -j NUM_JOBS
                        Maximum number of parallel workers in attacking
  --start_index START_INDEX, -s START_INDEX
                        Index of sample to start processing, used when you
                        want to restore progress
```

- With samples generated, you can use `train.py` / `test.py` to see the effect of adverserial perturbations.
  - FYI, the evaluation for training / testing should include the original data, and you can get a full dataset by simply concatenating (i.e., `cat` command on UNIX machines) normal data file and adverserial data file.

## Q3 Couterfactural Masking

This part first calculates importance of tokens in each sample using `integrated gradient` method, and then generate masked samples using the gradient information as tokens' weights.

Because the `tokens` here refers to tokens genrated by  `Transformers.BertTokenizer` (not the `token` in original sample), the training / testing process needs to be modified. Also, the couterfactural samples are natually labeled as `no_relation`  (**so `wiki80` dataset is excluded in this part**), the metrics used (originally as micro-F1) should be modified. Here you can use `train_cf.py` and `test_cf.py` to perform training and testing.

Let's use the scripts step by step:

- To generate file containing token with gradient information, run `Q3/integrated_gradient.py`:

```bash
$ python integrated_gradient.py -h
usage: integrated_gradient.py [-h] --input_file INPUT_FILE --model_path
                              MODEL_PATH --relation_path RELATION_PATH
                              [--output_file OUTPUT_FILE]
                              [--max_seq_len MAX_SEQ_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for loading weights of model to attack
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store token information with gradient
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
```

- To generate couterfactural samples based on gradient information, run `Q3/counterfactural.py`:

```bash
$ python counterfactual.py -h
usage: counterfactual.py [-h] --grad_file GRAD_FILE --input_file INPUT_FILE
                         --output_file OUTPUT_FILE [--keep_origin]
                         [--num_tokens NUM_TOKENS] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --grad_file GRAD_FILE, -g GRAD_FILE
                        Where the input file containing tokens with gradient
                        is
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store token information with gradient
  --keep_origin, -k     Whether to keep original samples(as tokenized samples)
  --num_tokens NUM_TOKENS, -n NUM_TOKENS
                        Maximum number of tokens masked(removed) in counter-
                        factual samples
  --threshold THRESHOLD, -t THRESHOLD
                        Lower bound of key words, i.e. the minimum required
                        weight of being a key word
```

- To perform training / testing on counterfactural datasets, you can run `Q3/train_cf.py` and `Q3/test_cf.py` in a manner just like using the `train.py` and `test.py`:

```bash
$ python train_cf.py -h
usage: train_cf.py [-h] --model_path MODEL_PATH [--restore] --train_path
                   TRAIN_PATH --valid_path VALID_PATH --relation_path
                   RELATION_PATH [--num_epochs NUM_EPOCHS]
                   [--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE]
                   [--metric {micro_f1,acc}]

optional arguments: ...  # Details omitted

$ python test_cf.py -h
usage: test_cf.py [-h] --model_path MODEL_PATH --test_path TEST_PATH
                  --relation_path RELATION_PATH [--max_seq_len MAX_SEQ_LEN]
                  [--batch_size BATCH_SIZE]

optional arguments: ...  # Details omitted
```

- The `Q3/fix_length.py` is a small tool script that aligns a given tokenized dataset to a fixed length:

```bash
$ python fix_length.py -h
usage: fix_length.py [-h] --input_file INPUT_FILE --output_file OUTPUT_FILE
                     [--max_seq_len MAX_SEQ_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        File containing tokenized dataset
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        File to output fixed dataset
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Number of tokens in a sample
```

## Q4 Debiased Masking

This part is designed to find tokens (words) that are unrelated but highly co-occurence with different relations.

You can generate auto rules (i.e., words that comes with a relation but has little to do with it semantically) using `Q4/generate_rules.py`, which is based on `TF-IDF` algorithm and finds tokens with higher scores for each relation.

After that, you may need to edit the rules finer-grained manually, to avoid mistakes (sometimes the words chosen are related with the relation's meaning, like `father/mother` in  `parent` relation, and masking the tokens would change the meaning of these sentences).

With rules derived above, you can generate debiased samples with `Q4/mask_by_rules.py`.

- Use `Q3/generate_rules.py` to automatically generate rules (***remember to check the rules manually before generating masked samples***):

```bash
$ python generate_rules.py -h
usage: generate_rules.py [-h] --input_file INPUT_FILE [INPUT_FILE ...]
                         --output_file OUTPUT_FILE [--num_rules NUM_RULES]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE [INPUT_FILE ...], -i INPUT_FILE [INPUT_FILE ...]
                        Where the input file containing dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store token information with gradient
  --num_rules NUM_RULES, -n NUM_RULES
                        Maximum number of key words chosen as rules for each
                        relation
```

- Use `Q4/mask_by_rules.py` to generate mask samples:

```bash
$ python mask_by_rules.py -h
usage: mask_by_rules.py [-h] --input_file INPUT_FILE --rule_file RULE_FILE
                        --output_file OUTPUT_FILE

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing dataset is
  --rule_file RULE_FILE, -r RULE_FILE
                        Where the json file containing substitution rule_dict
                        is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store token information with gradient
```

- After generating rules and testing samples, incorporate weighted sampling in training using `Q4/train_weighted.py`:

```bash
$ python train_weighted.py -h
usage: train_weighted.py [-h] --model_path MODEL_PATH [--restore] --train_path
                         TRAIN_PATH --valid_path VALID_PATH --relation_path
                         RELATION_PATH --rule_path RULE_PATH
                         [--num_epochs NUM_EPOCHS] [--max_seq_len MAX_SEQ_LEN]
                         [--batch_size BATCH_SIZE] [--metric {micro_f1,acc}]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH, -m MODEL_PATH
                        Full path for saving weights during training
  --restore             Whether to restore model weights from given model path
  --train_path TRAIN_PATH, -t TRAIN_PATH
                        Full path to file containing training data
  --valid_path VALID_PATH, -v VALID_PATH
                        Full path to file containing validation data
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --rule_path RULE_PATH, --rule RULE_PATH
                        Path to file containing rules for generating weights
                        of training samples
  --num_epochs NUM_EPOCHS, -e NUM_EPOCHS
                        Number of training epochs
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for training and testing
  --metric {micro_f1,acc}
                        Metric chosen for evaluation
```

## Q5 Semantic Bias

This part aims to test whether the model relies on some stereotypes in data or unnecessary links between some entities and relations. The experiment is inspired by Han et al. 2020, with some differences in our OE and ME settings. Also we form a debiased dataset to test model's ability of learning features from the context, not just from entities' names.

- Only-entity setting is to test model's ability when given only the names of entities, and the dataset could be generated using `Q5/only_entity.py`:

```bash
$ python only_entity.py -h
usage: only_entity.py [-h] --input_file INPUT_FILE [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where output moified dataset
```

- Mask-entity randomly setting is to test model's ability when given only the context (without the entities, but positional tokens are kept), and you can generate masked dataset using `Q5/mask_random.py`:

```bash
$ python mask_random.py -h
usage: mask_random.py [-h] --input_file INPUT_FILE [--output_file OUTPUT_FILE]
                      [--keep_origin] [--ratio RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where output moified dataset
  --keep_origin, -k     Whether to keep original samples in output dataset
  --ratio RATIO, -r RATIO
                        Ratio of random masking
```

- Mask-entity by entity frequency setting is to test how entities' frequency in samples affects model's ability to do Relation Classification. The dataset is generated with `Q5/mask_frequency.py`:

```bash
$ python mask_frequency.py -h
usage: mask_frequency.py [-h] --input_file INPUT_FILE
                         [--output_file OUTPUT_FILE] [--keep_origin]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where output moified dataset
  --keep_origin, -k     Whether to keep original samples in output dataset
```

- Select debiased dataset (***for testing***) by choosing samples whose `only_entity` version beat the Bert model using `select_debiased.py`:

```bash
$ python select_debiased.py -h
usage: select_debiased.py [-h] --input_file INPUT_FILE --original_data ORIGIN
                          --output_file OUTPUT_FILE --model_path MODEL_PATH
                          --relation_path RELATION_PATH
                          [--max_seq_len MAX_SEQ_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        Where the input file containing original dataset is
  --original_data ORIGIN, --origin ORIGIN
                        Where the input file containing original dataset is
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store token information with gradient
  --model_path MODEL_PATH, -m MODEL_PATH
                        Where the model for wrong prediction is
  --relation_path RELATION_PATH, -r RELATION_PATH
                        Full path to json file containing relation to index
                        dict
  --max_seq_len MAX_SEQ_LEN, -l MAX_SEQ_LEN
                        Maximum sequence length of bert model
```

- After data generation, you can run `train.py` or `test.py` to perform training / testing.

## How to Cite

```bibtex
@inproceedings{li2021robustness,
  title={On Robustness and Bias Analysis of BERT-Based Relation Extraction},
  author={Li, Luoqiu and Chen, Xiang and Ye, Hongbin and Bi, Zhen and Deng, Shumin and Zhang, Ningyu and Chen, Huajun},
  booktitle={China Conference on Knowledge Graph and Semantic Computing},
  pages={43--59},
  year={2021},
  organization={Springer}
}
```
