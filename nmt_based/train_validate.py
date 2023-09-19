import argparse, sys, os, random, json
from typing import Iterable, List
import torch
import torch.nn as nn
from model.seq2seq import Seq2SeqTransformer
from torch.utils.data import DataLoader
from utils.utils import *
from datetime import date, datetime
from timeit import default_timer as timer
from omegaconf import OmegaConf
from utils.m2_scorer.m2scorer import m2_score
from tqdm import tqdm
from nmt_based.model.dataset_wrapper import DatasetWrapper

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Your device is: ", DEVICE)
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--train_config', 
        type=str, 
        default='configs/nmt/train.yaml', 
        help='config file'
        )
    parser.add_argument(
        '--resume_model', 
        type=str, 
        default=False, 
        help='path to the checkpoint to resume training'
        )
    parser.add_argument(
        '--split', 
        type=str, 
        help='dataset to train on: boun or gec_tr',
        default='gec_tr'
        )
    parser.add_argument(
        '--save_all_ckpts',
        type=bool,
        default=False,
        help='save all checkpoints or not'
        )
    parser.add_argument(
        '--plot_loss_graphs',
        type=bool,
        default=False,
        help='plot and save loss graphs or not'
        )
    parser.add_argument(
        '--eval',
        type=bool,
        default=True,
        help='evaulate the model on evaluation set or not'
        )

    args = parser.parse_args()

    config = OmegaConf.load(args.train_config)
    exp_path_root = config.save_dir

    if config.experiment_name == '':
        config.experiment_name ='train_nmt_' + 'date_' + date.today().strftime("%m_%d") + '_time_' + datetime.now().strftime("%H_%M")

    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)

    checkpoint_dir = os.path.join(exp_path_root, config.experiment_name, 'checkpoints')
    logfile_dir = os.path.join(exp_path_root, config.experiment_name, 'log.txt')

    os.makedirs(exp_path_root, exist_ok=True)
    os.makedirs(os.path.join(exp_path_root, config.experiment_name), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    with open(os.path.join(exp_path_root, config.experiment_name ,"args.json"), "w") as f:
            args_to_save = OmegaConf.to_container(config)
            args_to_save["split"] = args.split
            json.dump(args_to_save, f)

    print(f"Training on {args.split} dataset.")

    dataset_wrapper = DatasetWrapper(split = args.split)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = dataset_wrapper.get_vocab_sizes()
    text_transform = dataset_wrapper.get_text_transform()
    

    def greedy_decode(model, src, src_mask, max_len, start_symbol):
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



    def train_epoch(model, optimizer, loss_fn, train_dataset, BATCH_SIZE):
        model.train()
        losses = 0
        train_iter = train_dataset
        train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=dataset_wrapper.collate_fn)
        
        for src, tgt in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()
        
        #scheduler.step()
        return losses / len(train_dataloader)

    @torch.no_grad()
    def evaluate(model, loss_fn, test_dataset, BATCH_SIZE):
        model.eval()
        losses = 0

        val_iter = test_dataset
        val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=dataset_wrapper.collate_fn)

        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :] #tokens


            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)


    @torch.no_grad()
    def translate(model: torch.nn.Module, src_sentence: str):
        model.eval()
        src = dataset_wrapper.text_transform[dataset_wrapper.SRC_LANGUAGE](src_sentence).view(-1, 1) #tokenization
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=dataset_wrapper.BOS_IDX).flatten()
        sentence = " ".join(dataset_wrapper.vocab_transform[dataset_wrapper.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("[BOS]", "").replace("[EOS]", "").replace(" ##", "")
        sentence = dataset_wrapper.tokenizer.decode(dataset_wrapper.tokenizer.encode(sentence)).replace("[CLS]", "").replace("[SEP]", "").strip()
        return sentence


    # start writing output to logfile

    
    transformer = Seq2SeqTransformer(config.model.n_encoding_layers, config.model.n_decoding_layers, config.model.emb_size, 
                                config.model.n_heads, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, config.model.ffn_hid_dim)
    print("Sequence to Sequence model initialized with the following parameters:")
    print_model_summary(config)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset_wrapper.PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.train.lr, betas=(config.train.beta1, config.train.beta2), 
                                eps=config.train.eps, weight_decay=config.train.weight_decay)
    if config.train.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.T_max)

    train_loss_history = []
    val_loss_history = []

    print_training_summary(config)
    for epoch in range(1, config.train.epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, dataset_wrapper.get_train_dataset(), config.train.batch_size)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, dataset_wrapper.get_val_dataset(), config.train.batch_size)

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if args.save_all_ckpts:
            logdir = os.path.join(exp_path_root, config.experiment_name, "checkpoints")
            torch.save(transformer.state_dict(), f'{logdir}/train_nmt_lr{config.train.lr}_epoch_{epoch}.pt')
            #save optimizer
            torch.save(optimizer.state_dict(), f'{logdir}/optimizer_nmt_lr{config.train.lr}_epoch_{epoch}.pt')
            print("Saving model and optimizer state at epoch {} to {}".format(epoch, logdir))
            print("=" * 33)

        if args.plot_loss_graphs:
            logdir_graph = os.path.join(exp_path_root, config.experiment_name, "loss_graphs")
            if epoch % 5 == 0:
                print("Plotting the loss graphs")
                print("=" * 33)
                plot_graphs(train_loss_history, val_loss_history, epoch, logdir_graph)


    if not args.save_all_ckpts:
        logdir = os.path.join(exp_path_root, config.experiment_name, "checkpoints")
        torch.save(transformer.state_dict(), f'{logdir}/train_nmt_lr{config.train.lr}_epoch_{epoch}.pt')
        #save optimizer
        torch.save(optimizer.state_dict(), f'{logdir}/optimizer_nmt_lr{config.train.lr}_epoch_{epoch}.pt')
        print("Finished training the model. Saving model and optimizer state at epoch {} to {}".format(epoch, logdir))
        print("=" * 33)


    if args.eval:
        with torch.no_grad():
            print("Evaluating the model on the evaluation set.")
            transformer.eval()
            source_sentences_eval = dataset_wrapper.source_sentences_val
            output = []
            with tqdm(total=len(source_sentences_eval)) as pbar:
                for s in source_sentences_eval:
                    output.append(translate(transformer, s))
                    pbar.update(1)

            #uncomment to get the output in a file
            # output_path = os.path.join(exp_path_root, config.exp_name, f"eval_pred_text.txt")
            # try:
            #     with open(output_path, "w") as f:
            #         for line in output:
            #             f.write(line + "\n")
            # except:
            #     print("Error writing to file")
                    
            p, r, f_05 = m2_score(output, split='val')
            #print p, r, f_05 in the log file
            print("Precision   : %.4f" % p, file=open(logfile_dir, "a"))
            print("Recall      : %.4f" % r, file=open(logfile_dir, "a"))
            print("F_%.1f       : %.4f" % (0.5, f_05), file=open(logfile_dir, "a"))

if __name__ == "__main__":
    main()
