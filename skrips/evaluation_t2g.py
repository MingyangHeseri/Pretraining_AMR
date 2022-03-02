from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments,Seq2SeqTrainingArguments
from transformers import Trainer,Seq2SeqTrainer
import random
import os
from datasets import load_metric
from ultils import *
import gc


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    sentence_test,amr_sequences_test = open_dataset(path = "/home/students/he/pythonProject1/test")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/pythonProject1/bart_model_g2t',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/pythonProject1/bart_model_g2t',local_files_only=True)
    amr_sequences_test = add_token(amr_sequences_test, sequence_type="amr")
    sentence_test = add_token(sentence_test, sequence_type="sentences")
    dataset_test = to_dataset(inpus_sequnce=sentence_test, labels_sequence=amr_sequences_test,tokenizer=tokenizer,
                         device=device)
    loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True)
    model.to(device)
    model.eval()
    decoded_sentences = []
    label_sentences = []
    with torch.no_grad():
        for batch in loader:
            generated_ids = model.generate(input_ids = batch["input_ids"], attention_mask = batch['attention_mask'],num_beams=5) # wenti chuxian
            decoded_sentences.append(generated_ids)
            label_sentences.append(batch["labels"])
    decoded_sentences = [item for sublist in decoded_sentences for item in sublist]
    label_sentences = [item for sublist in label_sentences for item in sublist]



# 在两个AMR 平价的时候需要把他们放到不同的文件当中






