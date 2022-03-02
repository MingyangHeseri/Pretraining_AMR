from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments
from transformers import Trainer
import random
import os
from ultils import *
import penman
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    path = "/home/students/he/pythonProject1/training"
    sentences,amr_sequences = open_dataset(path = path)
    print(amr_sequences[1])
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base') # load the bart model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    vocab_dic = tokenizer.get_vocab()  # get the vocabulary of bart and make it a dictionary
    all_amrs = " ".join(amr_sequences)
    token_set = set(all_amrs.split())  # get the token to be added
    token_to_add = []
    for token in token_set:
        if token not in vocab_dic.keys():
            token_to_add.append(token.lower())
    tokenizer.add_tokens(token_to_add)  # add the token to the tokenizer
    vocab_dic = tokenizer.get_vocab()
    model.resize_token_embeddings(len(tokenizer))  # resize the model
    new_amr_list_masked = masking_amr(amr_sequences,masking_rate = 0.15)
    amr_sequences = add_token(amr_sequences,sequence_type = "amr")
    new_amr_list_masked = add_token(new_amr_list_masked,sequence_type = "amr")
    dataset = to_dataset(inpus_sequnce = amr_sequences ,labels_sequence = new_amr_list_masked, tokenizer = tokenizer, device = device)
    model.to(device)
    model.train()
    args = TrainingArguments(
        output_dir='out',
        overwrite_output_dir='True',
        save_strategy = "epoch",
        dataloader_pin_memory=False,
        per_device_train_batch_size=4,
        num_train_epochs=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model("bart_model")
    tokenizer.save_pretrained("bart_model")
