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
import penman



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    sentence_test,amr_sequences_test = open_dataset(path = "/home/students/he/pythonProject1/test")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/pythonProject1/bart_model_direct_t2g',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/pythonProject1/bart_model_direct_t2g',local_files_only=True)
    amr_sequences_test = add_token(amr_sequences_test, sequence_type="amr")
    sentence_test = add_token(sentence_test, sequence_type="sentences")
    dataset_test = to_dataset(inpus_sequnce=sentence_test, labels_sequence=amr_sequences_test,tokenizer=tokenizer,
                         device=device)
    print(sentence_test[:5])
    loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True)
    model.to(device)
    model.eval()
    decoded_sentences = []
    label_sentences = []
    input_sentences = []
    with torch.no_grad():
        for batch in loader:
            generated_ids = model.generate(input_ids = batch["input_ids"], attention_mask = batch['attention_mask'],num_beams=5) # wenti chuxian

            input_sentence = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
            sentences_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
            decoded_sentences.append(sentences_decoded)
            label_sentences.append(decoded_labels)
            input_sentences.append(input_sentence)
    decoded_sentences = [item for sublist in decoded_sentences for item in sublist]
    label_sentences = [item for sublist in label_sentences for item in sublist]
    input_sentences = [item for sublist in input_sentences for item in sublist]
    print(decoded_sentences[:5])
    print(label_sentences[:5])
    print(input_sentences[:5])


