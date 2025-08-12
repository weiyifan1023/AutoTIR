import os
import datasets
import jsonlines
import argparse
import random
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./musique')

    args = parser.parse_args()

    train_data_path = os.path.join(args.local_dir, 'train.jsonl')
    lines = []
    with jsonlines.open(train_data_path) as reader:
        for line in reader:
            lines.append(line)
    train_data = []
    for line in lines:
        train_data.append({
            "data_source": "musique",
            "question": line['question'],
            "ability": "qa",
            "reward_model": {
                    "style": "rule",
                    "ground_truth": line['golden_answers']
                },
            "extra_info": {
                "id": line['id'],
            }
        })

    dev_data_path = os.path.join(args.local_dir, 'dev.jsonl')
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    dev_data = []
    random.shuffle(lines)
    for line in lines[:100]:
        dev_data.append({
            "data_source": "musique",
            "question": line['question'],
            "ability": "qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                "id": line['id'],
            }
        })

    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(dev_data)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))