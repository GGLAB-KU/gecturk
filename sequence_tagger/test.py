import sys
sys.path.append(".")
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import Dataset
import copy
import numpy as np
from utils.m2_scorer.m2scorer import m2_score
from data_generation.correction_rules import correction_rules
from data_utils import preprocess_input
import json
import os
import argparse
from constants import label_names
from datasets import load_metric


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: id for id, label in id2label.items()}

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--test_config', 
    type=str, 
    default='configs/sequence_tagger/test.yaml', 
    help='config file'
    )

args = parser.parse_args()

config = OmegaConf.load(args.test_config)

tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.test_dataset_dir, "r") as f:
        test_data = f.readlines()

    test_sentences, test_labels = preprocess_input(test_data)
    test_dataset = Dataset.from_dict({"tokens": test_sentences, "labels": test_labels})
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["tokens"])
    test_dataloader = torch.utils.data.DataLoader(
        tokenized_test_dataset,
        batch_size=config.test.batch_size,
        collate_fn=data_collator,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(config.model.model_name, 
                                                           num_labels=config.model.num_labels, 
                                                           id2label=id2label, 
                                                           label2id=label2id)
    
    model = model.to(device)
    
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in tqdm(test_dataloader):
        for key, value in batch.items():
            batch[key] = batch[key].to(device)


        with torch.no_grad():
            outputs = model(**batch)

        predictions = np.argmax(outputs["logits"].cpu(), axis=2)
        labels = batch["labels"]

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        all_predictions.extend(true_predictions)
        all_labels.extend(true_labels)
    
    metric = load_metric("seqeval")

    results = metric.compute(predictions=all_predictions, references=all_labels)
    
    n = len(test_sentences)
    model_outputs = copy.deepcopy(all_predictions)

    for idx in range(0, n):
        for i, label in enumerate(model_outputs[idx]):
            if label == "O":
                model_outputs[idx][i] = 0
            elif label == 0:
                continue
            else:
                correction_rule = correction_rules["rule_" + label.split("_")[-1]]
                try:
                    correction_rule(test_sentences[idx], model_outputs[idx], i) 
                except Exception as e:
                    print(e)
                    pass

    m2_results = m2_score([" ".join(sentence) for sentence in test_sentences], split="test",)

    results["m2_score_gectr"] = m2_results
    
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    with open(os.path.join(config.save_dir, config.experiment_name + ".json"), "w") as f:
        json.dump(results, f)
        
        
def tokenize_and_align_labels(data):
    tokenized_inputs = tokenizer(data["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length= 256)

    labels = []
    for i, label in enumerate(data["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    main()