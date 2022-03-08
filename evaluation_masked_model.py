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
     # Get the device
    sentence_test,amr_sequences_test=open_dataset(path = "/home/students/he/Pretraining_AMR/test")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/Pretraining_AMR/bart_model',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/Pretraining_AMR/bart_model',local_files_only=True)
    new_amr_list_masked = masking_amr(amr_sequences_test, masking_rate=0.15)
    amr_sequences_test = add_token(amr_sequences_test, sequence_type="amr")
    new_amr_list_masked = add_token(new_amr_list_masked, sequence_type="amr")
    dataset_test = to_dataset(inpus_sequnce= new_amr_list_masked, labels_sequence=amr_sequences_test,tokenizer=tokenizer,
                         device=device)
    # tokenizer1 = BartTokenizer.from_pretrained('facebook/bart-base')
    # print("tokenizer_test")
    # text = "i had a relationship with a 16 year old boy and was forbidden from spea"
    # tokenizer_test = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length',
    #                    add_special_tokens=False).to(device)
    # tokenizer_test1 = tokenizer1(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length',
    #                    add_special_tokens=False).to(device)
    # print(tokenizer1.batch_decode(tokenizer_test1["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    # print(tokenizer.batch_decode(tokenizer_test["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True))

    loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True)
    model.to(device)
    model.eval()
    decoded_sentences = []
    label_sentences = []
    input_sentences = []
    with torch.no_grad():
        for batch in loader:
            generated_ids = model.generate(input_ids = batch["input_ids"], attention_mask = batch['attention_mask'],num_beams=5,length_penalty = 1.0)
            decoded_labels = tokenizer.batch_decode(batch["labels"],
                                                    clean_up_tokenization_spaces=True)
            input_sentence = tokenizer.batch_decode(batch["input_ids"],
                                                    clean_up_tokenization_spaces=True)
            sentences_decoded = tokenizer.batch_decode(generated_ids,
                                                       clean_up_tokenization_spaces=True)
            decoded_sentences.append(sentences_decoded)
            label_sentences.append(decoded_labels)
            input_sentences.append(input_sentence)
    decoded_sentences = [item for sublist in decoded_sentences for item in sublist]
    label_sentences = [item for sublist in label_sentences for item in sublist]
    input_sentences = [item for sublist in input_sentences for item in sublist]
    reference_sentences = []
    for label in label_sentences:
        reference_sentences.append([label])
    print(decoded_sentences[:5])
    print(label_sentences[:5])
    print(input_sentences[:5])
