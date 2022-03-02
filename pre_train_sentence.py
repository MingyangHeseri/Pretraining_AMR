from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments
from transformers import Trainer
import random
import os
from transformers import DataCollatorForLanguageModeling
from ultils import *
import penman
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    path = "/home/students/he/pythonProject1/training"
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    sentences, amr_sequences = open_dataset(path=path)
    dataset = mask_sentence(tokenizer=tokenizer,sentences=sentences,masking_rate=0.15,device=device)
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
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("bart_model_sentence")
    tokenizer.save_pretrained("bart_model_sentence")


