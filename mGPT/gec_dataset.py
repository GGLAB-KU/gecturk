from torch.utils.data import Dataset
import torch
import numpy as np
import re
from openprompt.data_utils import InputExample

class GECDataset(Dataset):
    def __init__(self, source_dataset_path, tgt_dataset_path, evaluation=False, detection=False):
        self.evaluation = evaluation
        self.detection = detection
        self.source_text, self.target_text, self.problem_rules, self.span_pairs, self.solution_words = self._load_dataset(source_dataset_path, tgt_dataset_path)
        print(len(self.source_text), len(self.target_text))
        
        self.problem_rules = list(map(lambda x: int(x.split("_")[1]), self.problem_rules))
        
        self.count = 0
    
    def __len__(self):
        return len(self.source_text)
        
    def _load_dataset(self, source_dataset_path, tgt_dataset_path):
        source_text = []
        target_text = []
        problem_rules = []
        span_pairs = []
        solution_words = []
        
        with open(source_dataset_path, "r") as f:
            raw_data_source = f.read()
            elements = raw_data_source.split("\n\n")
        with open(tgt_dataset_path, "r") as f:
            raw_data_tgt = f.readlines()
        for element, tgt in zip(elements, raw_data_tgt):
            element = element.rstrip()
            lines = element.split("\n")
            lines = list(filter(('').__ne__, lines))
            if len(lines) == 0:
                continue

            num_options = len(lines) - 1
            source = lines[0][2:]

            if num_options == 0 or self.evaluation:
                source_text.append(source.strip())
                target_text.append(tgt.strip())
                problem_rules.append("rule_0")
                span_pair_start = -1
                span_pair_end = -1
                span_pairs.append((span_pair_start, span_pair_end))
                solution_words.append("")
            else:
                for i in range(num_options):
                    #option = lines[len(lines) - i - 1][2:]
                    option = lines[len(lines) - i - 1]
                    option = option[2:]
                    option_categories = option.split("|||")
                    source_text.append(source.strip())
                    #target_text.append(target)
                    target_text.append(tgt.strip())
                    problem_rules.append(option_categories[1])
                    span_pair_start = option_categories[0].split(" ")[0]
                    span_pair_end = option_categories[0].split(" ")[1]
                    span_pairs.append((span_pair_start, span_pair_end))
                    solution_words.append(option_categories[2])
        source_text = list(filter(lambda x: x != "\n", source_text))
        target_text = list(filter(lambda x: x != "\n", target_text))
        
        return source_text, target_text, problem_rules, span_pairs, solution_words
    
    def __getitem__(self, i):
        input_example = InputExample(text_a = self.source_text[i], label = self.problem_rules[i], guid=i, tgt_text=self.target_text[i].strip())
        return input_example
    
    def __iter__(self):
        examples = []
        for i in range(self.__len__()):
            # For generation
            if self.detection:
                input_example = InputExample(text_a = self.source_text[i].strip(), label = self.problem_rules[i], guid=i, tgt_text= self.target_text[i].strip() + "\n" + ";".join([str(self.problem_rules[i]), str(self.span_pairs[i][0]), str(self.span_pairs[i][1])]))
            else:
                input_example = InputExample(text_a = self.source_text[i], label = self.problem_rules[i], guid=i, tgt_text=self.target_text[i])

            examples.append(input_example)
        return iter(examples)
