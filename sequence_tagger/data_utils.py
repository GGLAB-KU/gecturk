from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_metric
import torch
import numpy as np
from constants import label_names

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

def preprocess_input(data):

    source_sentences, tags = parse_input(data)
        
    labels = []
    split_sentence = []

    sentence_len = len(source_sentences)

    for i in range(sentence_len):
        sentence = source_sentences[i].strip().split(" ")
        sentence = [word for word in sentence if word] # remove empty strings
        label = [0]*len(sentence)

        for tag in tags[i]:
            annotation = Annotation(tag)

            for idx in annotation.get_indices():
                label[idx] = annotation.get_tag_number()

        labels.append(label)
        split_sentence.append(sentence)

    return split_sentence, labels

def model_init():
    return AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-turkish-cased", 
                                                           num_labels=26, 
                                                           id2label=id2label, 
                                                           label2id=label2id)

def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if(k not in flattened_results.keys()):
            flattened_results[k+"_f1"]=results[k]["f1"]

    return flattened_results

def parse_input(data):
    source_sentences = []
    tags = []

    for line in data:    
        if line.startswith("S"):
            current_tags = []
            source_sentences.append(line[2:])
        elif line.startswith("A"):
            current_tags.append(line)
        elif line == "\n":
            tags.append(current_tags)

    tags.append(current_tags)
    
    return source_sentences, tags

class Annotation:
    def __init__(self, tag):
        tag = tag.split("|||")
        self.starting_index = int(tag[0].split(" ")[1])
        self.ending_index = int(tag[0].split(" ")[2])
        self.rule_name = tag[1]
        self.correction = tag[2]
        
    def get_indices(self):
        return list(range(self.starting_index, self.ending_index)) 
    
    def get_tag_number(self):
        return int(self.rule_name.split("_")[-1])