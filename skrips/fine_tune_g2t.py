from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments,Seq2SeqTrainingArguments
from transformers import Trainer,Seq2SeqTrainer
import random
import os
from datasets import load_metric
from ultils import *


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    sentences,amr_sequences = open_dataset(path = "/home/students/he/pythonProject1/training")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/pythonProject1/bart_model_t2g',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/pythonProject1/bart_model_t2g',local_files_only=True)
    sentences = add_token(sentences,sequence_type = "sentence")
    amr_sequences = add_token(amr_sequences,sequence_type = "amr")
    dataset = to_dataset(inpus_sequnce=amr_sequences, labels_sequence=sentences, tokenizer=tokenizer,
                         device=device)
    dataset_test = to_dataset(inpus_sequnce=amr_sequences_test, labels_sequence=sentence_test,tokenizer=tokenizer,
                         device=device)
    model.to(device)
    model.train()
    args = Seq2SeqTrainingArguments(
        output_dir='out',
        overwrite_output_dir='True',
        save_strategy = "epoch",
        dataloader_pin_memory=False,
        per_device_train_batch_size=4,
        num_train_epochs=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model("bart_model_g2t")
    tokenizer.save_pretrained("bart_model_g2t")








