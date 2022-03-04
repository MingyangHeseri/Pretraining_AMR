from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments,Seq2SeqTrainingArguments
from transformers import Trainer,Seq2SeqTrainer
import random
import os
from datasets import load_metric
from ultils import *
import penman


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
     # Get the device
    path = "/home/students/he/Pretraining_AMR/training"
    sentences,amr_sequences = open_dataset(path = path)
    print(amr_sequences[1])
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base') # load the bart model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    vocab_dic = tokenizer.get_vocab()  # get the vocabulary of bart and make it a dictionary
    #all_amrs = " ".join(amr_sequences)
    #token_set = set(all_amrs.split())  # get the token to be added
    token_to_add = []
    with open("vocab.txt", "r") as outfile:
        vocab_list = outfile.read().splitlines()
    for token in vocab_list:
        if token not in vocab_dic.keys():
            token_to_add.append(token.lower())
    tokenizer.add_tokens(token_to_add)  # add the token to the tokenizer
    vocab_dic = tokenizer.get_vocab()
    model.resize_token_embeddings(len(tokenizer))  # resize the model
    amr_sequences = add_token(amr_sequences, sequence_type="amr") # add token
    sentences = add_token(sentences, sequence_type="sentence")
    print(sentences[:5]) # see the input
    print(amr_sequences[:5])
    dataset = to_dataset(inpus_sequnce=amr_sequences, labels_sequence=sentences, tokenizer=tokenizer,
                         device=device)
    print(device)
    model.to(device)
    model.train()
    args = Seq2SeqTrainingArguments(
        output_dir='out',
        overwrite_output_dir='True',
        save_strategy = "epoch",
        learning_rate = 3e-05,
        dataloader_pin_memory=False,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        fp16 = True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model("bart_model_direct_g2t")
    tokenizer.save_pretrained("bart_model_direct_g2t")