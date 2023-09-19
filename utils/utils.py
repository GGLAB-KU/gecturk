import matplotlib.pyplot as plt
import torch
import json
from omegaconf import DictConfig, OmegaConf

def read_data(sPATH, tPATH):
    
    with open(sPATH, "r") as f:
        data = f.readlines()

    source_sentence = []
    tags = []
    target_sentences = []
    
    for line in data:    
        if line.startswith("S"):
            current_tags = []
            source_sentence.append(line[2:])
        elif line.startswith("A"):
            current_tags.append(line)
        elif line == "\n":
            tags.append(current_tags)

    tags.append(current_tags)
    if tPATH is not None:
        with open(tPATH, "r") as f:
            target_sentences = [line.strip() for line in f.readlines()]
            target_sentences = [a for a in target_sentences if a != '']
    source_sentence = [line.strip() for line in source_sentence]
    return source_sentence, target_sentences


def plot_graphs(train_losses, val_losses, epochs, logdir):
    plt.plot(range(1,epochs+1), train_losses, label='Training loss')
    plt.plot(range(1,epochs+1), val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #save graph
    try:         
        plt.savefig(f'{logdir}/train_validation_loss_{epochs}.png')
        plt.show()
    
    except:
        print("unable to save the image")
        plt.show()

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device='cuda').type(torch.bool)

    PAD_IDX = 0
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def print_model_summary(config):
    print("Model Summary")
    print(f"Number of Encoder Layers: {config.model.n_encoding_layers}")
    print(f"Number of Decoder Layers: {config.model.n_decoding_layers}")
    print(f"Embedding Size: {config.model.emb_size}")
    print(f"Number of Heads: {config.model.n_heads}")
    print(f"Feed Forward Hidden Dimension: {config.model.ffn_hid_dim}")
    print("=" * 33)

def print_training_summary(config):
    print("Started Training the Model")
    print(f"Number of Epochs: {config.train.epochs}. Batch Size: {config.train.batch_size}. Learning Rate: {config.train.lr}")
    print("=" * 33)

    
class GECDataset(torch.utils.data.Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]

def get_config_from_json(json_path):

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    OmegaConf.register_new_resolver("load_json", load_json)
    cfg = OmegaConf.create({"exp": OmegaConf.create(load_json(json_path))})
    return cfg