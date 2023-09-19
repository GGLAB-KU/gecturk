import sys
sys.path.append(".")
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from omegaconf import OmegaConf
import os
import argparse
from datasets import Dataset
from constants import label_names
from data_utils import preprocess_input, model_init, compute_metrics

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: id for id, label in id2label.items()}

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--train_config', 
    type=str, 
    default='configs/sequence_tagger/train.yaml', 
    help='config file'
    )

args = parser.parse_args()

config = OmegaConf.load(args.train_config)

tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)

def main():

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    #Load data
    with open(config.train_dataset_dir, "r") as f:
        training_data = f.readlines()
    
    with open(config.val_dataset_dir, "r") as f:
        validation_data = f.readlines()
    
    training_sentences, training_labels = preprocess_input(training_data)
    val_sentences, val_labels = preprocess_input(validation_data)

    training_dataset = Dataset.from_dict({"tokens": training_sentences, "labels": training_labels})
    val_dataset = Dataset.from_dict({"tokens": val_sentences, "labels": val_labels})
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    tokenized_train_dataset = training_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
    output_dir=config.save_dir,
    per_device_train_batch_size=config.train.batch_size,
    per_device_eval_batch_size=config.train.batch_size,
    num_train_epochs=config.train.num_of_epochs,
    weight_decay=config.train.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config.train.learning_rate,
    load_best_model_at_end=True,
    seed=config.train.seed
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset= tokenized_train_dataset,
        eval_dataset= tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    trainer.save_model(os.path.join(config.save_dir, "best_model"))

def tokenize_and_align_labels(data):
    tokenized_inputs = tokenizer(data["tokens"], truncation=True, is_split_into_words=True)

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