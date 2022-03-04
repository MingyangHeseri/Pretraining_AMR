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
    sentences,amr_sequences = open_dataset(path = "/home/students/he/Pretraining_AMR/training")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/Pretraining_AMR/bart_model',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/Pretraining_AMR/bart_model',local_files_only=True)
    new_amr_list_masked = mask_subgraph(amr_sequences, masking_rate=0.15)
    amr_sequences = add_token(amr_sequences,sequence_type = "amr")
    new_amr_list_masked = add_token(new_amr_list_masked,sequence_type = "amr")
    dataset = to_dataset(inpus_sequnce=new_amr_list_masked, labels_sequence=amr_sequences, tokenizer=tokenizer,
                         device=device)
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
    trainer.save_model("bart_model_subgraph")
    tokenizer.save_pretrained("bart_model_subgraph")
