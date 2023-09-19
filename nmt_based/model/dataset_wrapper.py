from utils.utils import *
from transformers import AutoTokenizer
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence

class DatasetWrapper():
    def __init__(self, split):

        if split == 'boun':
            train_src_path = 'datasets/boun/boun_source_train.txt'
            train_tgt_path = 'datasets/boun/boun_target_train.txt'
            val_src_path = 'datasets/boun/boun_source_dev.txt'
            val_tgt_path = 'datasets/boun/boun_target_dev.txt'
            test_src_path = 'datasets/boun/boun_source_test.txt'
            test_tgt_path = 'datasets/boun/boun_target_test.txt'

        elif split == 'gec_tr':
            train_src_path = 'datasets/gec_tr/gec_tr_source_annot_train.txt'
            train_tgt_path = 'datasets/gec_tr/gec_tr_target_train.txt'
            val_src_path = 'datasets/gec_tr/gec_tr_source_annot_val.txt'
            val_tgt_path = 'datasets/gec_tr/gec_tr_target_val.txt'
            test_src_path = 'datasets/gec_tr/gec_tr_source_annot_test.txt'
            test_tgt_path = 'datasets/gec_tr/gec_tr_target_test.txt'
   
            self.SRC_LANGUAGE = 'incorrect'
            self.TGT_LANGUAGE = 'correct'
            self.train_src_path = train_src_path
            self.train_tgt_path = train_tgt_path
            self.source_sentences_train, self.target_sentences_train = read_data(train_src_path, train_tgt_path)
            self.source_sentences_val, self.target_sentences_val = read_data(val_src_path, val_tgt_path)
            self.source_sentences_test, self.target_sentences_test = read_data(test_src_path, test_tgt_path)
            
            self.train_dataset = GECDataset(self.source_sentences_train, self.target_sentences_train)
            self.val_dataset = GECDataset(self.source_sentences_val, self.target_sentences_val)
            
            # Place-holders
            self.text_transform = {}
            self.token_transform = {}
            self.vocab_transform = {}
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

            self.token_transform[self.SRC_LANGUAGE] = self.tokenizer
            self.token_transform[self.TGT_LANGUAGE] = self.tokenizer
            self.SRC_VOCAB_SIZE = 0
            self.TGT_VOCAB_SIZE = 0

            # Define special symbols and indices
            self.PAD_IDX, self.UNK_IDX, self.SEP_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3, 4 
            # Make sure the tokens are in order of their indices to properly insert them in vocab
            self.special_symbols = ['[PAD]', '[UNK]', '[SEP]', '[BOS]', '[EOS]']
    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.SRC_LANGUAGE: 0, self.TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield self.token_transform[language].tokenize(data_sample[language_index[language]])

    def get_vocab_transform(self):
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            # Training data Iterator
            train_iter = self.train_dataset
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                min_freq=1,
                                                                specials=self.special_symbols,
                                                                special_first=True
                                                            )

        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

        self.SRC_VOCAB_SIZE = len(self.vocab_transform[self.SRC_LANGUAGE])
        self.TGT_VOCAB_SIZE = len(self.vocab_transform[self.TGT_LANGUAGE])

        return self.vocab_transform

    def get_vocab_sizes(self):
        self.get_vocab_transform()
        return self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE


    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_test_dataset(self):
        return self.source_sentences_test
    # helper function to club together sequential operations
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
    

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]), 
                        torch.tensor(token_ids), 
                        torch.tensor([self.EOS_IDX])))

    def tokenize_helper(self, language):
        def tokenize(text):
            return self.token_transform[language].tokenize(text)
        return tokenize
    
    def get_text_transform(self):
        # src and tgt language text transforms to convert raw strings into tensors indices
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.text_transform[ln] = self.sequential_transforms(self.tokenize_helper(ln), #Tokenization
                                                    self.vocab_transform[ln], #Numericalization
                                                    self.tensor_transform)
        
        return self.text_transform
    
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(torch.tensor(self.text_transform[self.SRC_LANGUAGE](src_sample.rstrip("\n")).clone().detach()))
            tgt_batch.append(torch.tensor(self.text_transform[self.TGT_LANGUAGE](tgt_sample.rstrip("\n")).clone().detach()))
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch