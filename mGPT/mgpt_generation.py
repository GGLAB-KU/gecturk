import torch
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
from openprompt.plms import load_plm, add_special_tokens
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from openprompt.data_utils import InputExample
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from openprompt.utils.metrics import generation_metric
from tqdm import tqdm
import gec_dataset
import gec_dataset_detection
import re
import argparse
import random

# Parser options
parser = argparse.ArgumentParser(
                    prog = 'Train M-GPT Token Classification',
                    description = 'Trains M-GPT on the dataset for token classification and produces evaluation',
                    epilog = '')

parser.add_argument("--initial_lr", default=5e-5, help="Starting learning rate for the linear scheduler")
parser.add_argument("--model_name", default="gpt2", help="Base model configuration (not the specific weights/trained model)")
parser.add_argument("--detection", default=False, type=bool, help="Whether to try correction + detection, or just correction")
parser.add_argument("--model_path", default="ai-forever/mGPT", help="The huggingface name of the model")
parser.add_argument("--dataset_percentage", required=True, type=float, help="What percentage of the training data to use during training? (Used for ablation analysis)")
parser.add_argument("--batch_size", default=3, type=int)
parser.add_argument("--epochs", required=True, type=int)
parser.add_argument("--weights_file", required=True, help="Where to save the weights after training finishes?")
parser.add_argument("--num_beams", default=5, type=int, help="How many beams to use during beam search for generation?")
parser.add_argument("--mixed_precision", default=False, help="Whether to use full precision or mixed precision during training and generation.")
parser.add_argument("--output_file", required=True, help="Where to output the test results for metric calculations.")
parser.add_argument("--resume_weights", default=None, help="If needed, path to model weights, to resume training from a previous checkpoint.")
parser.add_argument("--seed", default=0, help="Random seed to use for reproducability")
parser.add_argument("--just_eval", default=False, help="Random seed to use for reproducability")

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

lr = args.initial_lr
plm_eval_mode = True
model_name = args.model_name
model_name_or_path = args.model_path

# Using the load_plm function provided by OpenPrompt to load the model
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_name_or_path)
# The tokenizer for m-gpt does not contain a padding token, so set it to be the same as the eos token
if args.model_name == "gpt2":
    tokenizer.pad_token = tokenizer.eos_token
elif args.model_name == "t5":
    tokenizer.additional_special_tokens = ["<pad>"]
    tokenizer.additional_special_tokens_ids = [0]

# Loading the datasets. These are currently hard-coded but can be modified in the future as needed.
dataset = {}
# Here, we only keep a certain percentage of the training dataset
train_dataset = list(iter(gec_dataset.GECDataset(f"./dataset_splits/gec_src_train_{int(args.dataset_percentage * 100)}.txt", f"./dataset_splits/gec_tgt_train_{int(args.dataset_percentage * 100)}.txt")))

src_train_path = "./datasets/gec_tr/gec_tr_source_annot_train.txt"
tgt_train_path = "./datasets/gec_tr/gec_tr_target_train.txt"
src_test_path = "./datasets/gec_tr/test_sentences_final.txt"
tgt_test_path = "./datasets/gec_tr/test_sentences_final.txt"

if args.detection:
    train_dataset = list(iter(gec_dataset_detection.GECDataset(src_train_path, tgt_train_path)))
else:
    train_dataset = list(iter(gec_dataset.GECDataset(src_train_path, tgt_train_path)))

test_dataset = list(iter(gec_dataset.GECDataset(src_test_path, src_test_path, evaluation=True)))
# We use prefix tuning here, so we need this template. The text is the standard template for text provided by OpenPrompt
mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder": "text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# Creating the different dataloaders. The training dataloader needs both shuffling and teacher_forcing for the training process, but for proper evaluation, they are disabled for the validation and test dataloaders. 
train_dataloader = PromptDataLoader(dataset=train_dataset, template=mytemplate, tokenizer=tokenizer, 
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, 
                                    decoder_max_length=512, batch_size=args.batch_size, shuffle=True, teacher_forcing=True,
                                    predict_eos_token=True, truncate_method="head")

test_dataloader = PromptDataLoader(dataset=test_dataset, template=mytemplate, tokenizer=tokenizer, 
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=350, 
                                    decoder_max_length=256, batch_size=1, shuffle=False, teacher_forcing=False,
                                    predict_eos_token=True, truncate_method="head")

use_cuda = True
# We are not fine-tuning the model itself, so we must specify that we freeze the plm
prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer, 
                                   plm_eval_mode=plm_eval_mode)

# Option to resume from a previous checkpoint if necessary
if args.resume_weights is not None:
    prompt_model.load_state_dict(torch.load(args.resume_weights))
if use_cuda:
    prompt_model = prompt_model.cuda()
    
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

# Using the recommended AdamW optimizer for training, with recommended eps value
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

# Use a linear weight schedule for the optimizer
tot_step = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

"""
Arguments used during generation.
---------------------------------
    - max_length: The maximum number of tokens that the model can generate.
    - max_new_tokens: Max length ignoring the number of tokens already in the prompt.
    - min_length: Minimum number of tokens required to be generated.
    - temperature: Used to module next token probabilities.
    - do_sample: If true, use sampling. Otherwise, uses greedy decoding.
    - top_k: How many highest probability tokens to keep for top-k filtering.
    - top_p: Only set of tokens with probability adding up to >= top_p are kept for generation.
    - repetition_penalty: Used to punish repetitions. 1.0 means no penalty. Follows the work of https://arxiv.org/pdf/1909.05858.pdf
    - num_beams: Number of beams to use during the beam search. If equal to 1, beam search is not used.
    - bad_words_ids: List of tokens that are not allowed to be generated.

"""
generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": args.num_beams,
    "bad_words_ids": [[628], [198]]
}

global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(args.epochs):
    prompt_model.train()
    # Typical PyTorch training loop, with gradient clipping
    for step, inputs in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        global_step +=1
        if args.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if use_cuda:
                    inputs = inputs.cuda()
                loss = prompt_model(inputs)
                loss.backward()
                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if global_step % 500 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss
            torch.save(prompt_model.state_dict(), args.weights_file)
# After training, save model weights.
if not args.just_eval:
    torch.save(prompt_model.state_dict(), args.weights_file)

#Generating the text file for metrics calculation. For M-GPT, very time-consuming.
validation_output = open(args.output_file, "w")
with torch.no_grad():
    generated_sentence = []
    prompt_model.eval()
    if args.mixed_precision:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for data in tqdm(test_dataloader, total=len(test_dataloader)):
                data = data.cuda()
                #print(data)
                # Here, we use the generation parameters listed above. prompt_model.generate is a wrapper for transformers.GenerationMixin.generate()
                _, output_sentence = prompt_model.generate(data, max_length=len(data.input_ids[0]) + 20, 
                                                        max_new_tokens=None, min_length=mytemplate.num_token, 
                                                        temperature=1.0, do_sample=False, 
                                                        top_k=0, top_p=0.9, 
                                                        repetition_penalty=0.5, num_beams=args.num_beams)#, 
                                                        #bad_word_ids=[[628], [198]])

                for sentence in output_sentence:
                    validation_output.write(sentence + "\n\n")
                    validation_output.flush()
    else:
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            data = data.cuda()
            # Here, we use the generation parameters listed above. prompt_model.generate is a wrapper for transformers.GenerationMixin.generate()
            _, output_sentence = prompt_model.generate(data, max_length=400, 
                                                    max_new_tokens=None, min_length=mytemplate.num_token, 
                                                    temperature=1.0, do_sample=False, 
                                                    top_k=0, top_p=0.9, 
                                                    repetition_penalty=0.5, num_beams=args.num_beams) 
                                                    #bad_word_ids=[[628], [198]])

            for sentence in output_sentence:
                validation_output.write(sentence + "\n\n")
                validation_output.flush()
        
validation_output.close()
