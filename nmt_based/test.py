import argparse, sys, os, random, json
from model.seq2seq import Seq2SeqTransformer
sys.path.append(".")
from utils.utils import *
from datetime import date, datetime
from timeit import default_timer as timer
from utils.m2_scorer.m2scorer import m2_score
from tqdm import tqdm
from nmt_based.model.dataset_wrapper import DatasetWrapper

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Your device is: {DEVICE}")
print("="*33)
# actual function to translate input sentence into target language

def greedy_decode(model, src, src_mask, max_len, start_symbol, dataset_wrapper):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == dataset_wrapper.EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str, dataset_wrapper: DatasetWrapper):
    model.eval()
    src = dataset_wrapper.text_transform[dataset_wrapper.SRC_LANGUAGE](src_sentence).view(-1, 1) #tokenization
    num_tokens = src.shape[0]
    src.to(DEVICE)
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=dataset_wrapper.BOS_IDX, dataset_wrapper=dataset_wrapper).flatten()
    sentence = " ".join(dataset_wrapper.vocab_transform[dataset_wrapper.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("[BOS]", "").replace("[EOS]", "").replace(" ##", "")
    sentence = dataset_wrapper.tokenizer.decode(dataset_wrapper.tokenizer.encode(sentence)).replace("[CLS]", "").replace("[SEP]", "").strip()
    return sentence


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--test_config', 
        type=str, 
        default='configs/nmt/test.yaml', 
        help='config file'
        )
    parser.add_argument(
        '--split', 
        type=str, 
        default='gec_tr',
        help='dataset to train on: boun or gec_tr',
        )
    parser.add_argument(
        '--save_pred_samples', 
        type=bool, 
        default=True,
        help='whether to save the output or not',
        
        )
    parser.add_argument(
        '--ckpt_dir', 
        type=str, 
        default='weights/nmt/nmt_gec_lr_1e4_epoch_120_best.pt', 
        help='config file'
        )
    

    args = parser.parse_args()

    config = OmegaConf.load(args.test_config)
    exp_path_root = 'experiments'
    
    if config.experiment_name == '':
        print("Testing on default pre-trained nmt model.")
        ckpt_dir = args.ckpt_dir
        config.experiment_name = 'default_nmt_test_results'
        os.makedirs(os.path.join(exp_path_root, config.experiment_name), exist_ok=True)
        model_args = config
        
    else:
        print(f"Testing on ckpt saved at {os.path.join(exp_path_root, config.experiment_name)}.")
        ckpt_dir = os.path.join(exp_path_root, config.experiment_name, 'checkpoints', config.ckpt_name)
        #read model args
        model_args = get_config_from_json(os.path.join(exp_path_root, config.experiment_name, 'args.json')).exp

    log_eval = os.path.join(exp_path_root, config.experiment_name, 'results.txt')
    #setup datase wrapper

    dataset_wrapper = DatasetWrapper(split = args.split)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = dataset_wrapper.get_vocab_sizes()
    text_transform = dataset_wrapper.get_text_transform()
    test_data = dataset_wrapper.get_test_dataset()

    #load model
    test_model = Seq2SeqTransformer(model_args.model.n_encoding_layers, model_args.model.n_decoding_layers, model_args.model.emb_size, 
                                model_args.model.n_heads, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, model_args.model.ffn_hid_dim)
    test_model.load_state_dict(torch.load(ckpt_dir))
    test_model.to(DEVICE)
    test_model.eval()
    print("Model loaded.")
    print_model_summary(config)

    output = []
    with torch.no_grad():
        with tqdm(total=len(test_data)) as pbar:
            for i, s in enumerate(test_data):
                output.append(translate(test_model, s, dataset_wrapper=dataset_wrapper))
                if i == 0:
                    print("Sample translation: ")
                    print("Input: ", s)
                    print("Output: ", output[-1])
                pbar.update(1)

        if args.save_pred_samples:
            output_path = os.path.join(exp_path_root, config.experiment_name, f"{config.experiment_name}_pred_samples.txt")
            with open(output_path, "w") as f:
                for line in output:
                    f.write(line + "\n")

        p, r, f_05 = m2_score(output, split = "test")
        #print p, r, f_05 in the log file
        print("Precision   : %.4f" % p, file=open(log_eval, "a"))
        print("Recall      : %.4f" % r, file=open(log_eval, "a"))
        print("F_%.1f       : %.4f" % (0.5, f_05), file=open(log_eval, "a"))


if __name__ == "__main__":
    main()




    
    

            
        
        
    
    
    
