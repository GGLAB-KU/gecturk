from torch.utils.data import Dataset
import torch
import numpy as np
import re
from openprompt.data_utils import InputExample
from tqdm import trange

class GECDataset(Dataset):
    def __init__(self, source_dataset_path, tgt_dataset_path, evaluation=False, detection=False):
        self.evaluation = evaluation
        self.detection = detection
        self.source_text, self.target_text, self.problem_rules, self.span_pairs, self.solution_words, self.for_detection = self._load_dataset(source_dataset_path, tgt_dataset_path)
        self.count = 0
    
    def __len__(self):
        return len(self.source_text)
        
    def _load_dataset(self, source_dataset_path, tgt_dataset_path):
        source_text = []
        target_text = []
        problem_rules = []
        span_pairs = []
        solution_words = []

        for_detection = []
        
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
            #num_options = len(lines) - 2
            num_options = len(lines) - 1
            source = lines[0][2:]
            #target = lines[1][2:]

            #if num_options == 0:
            #    continue
            #else:
            source_text.append(source)
            target_text.append(tgt)
            detection = []
            for i in range(num_options):
                option = lines[len(lines) - i - 1]
                option = option[2:]
                option_categories = option.split("|||")
                detection.append((int(option_categories[1].split("_")[1]), option_categories[0].split(" ")[0], option_categories[0].split(" ")[1]))
            for_detection.append(detection)


                            
        
        return source_text, target_text, problem_rules, span_pairs, solution_words, for_detection
    
    def __getitem__(self, i):
        input_example = InputExample(text_a = self.source_text[i], label = self.problem_rules[i], guid=i, tgt_text=self.target_text[i])
        return input_example
    
    def __iter__(self):
        examples = []
        for i in trange(self.__len__()):
            target_text = self.target_text[i].strip()

            for row in self.for_detection[i]:
                target_text = target_text.strip() + "\n" + ";".join(list(map(lambda x: str(x), row)))

            input_example = InputExample(text_a = self.source_text[i], guid=i, tgt_text = target_text)
            examples.append(input_example)
        return iter(examples)