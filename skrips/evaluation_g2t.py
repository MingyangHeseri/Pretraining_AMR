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
    sentence_test,amr_sequences_test=open_dataset(path = "/home/students/he/pythonProject1/test")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/pythonProject1/bart_model_g2t',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/pythonProject1/bart_model_g2t',local_files_only=True)
    amr_sequences_test = add_token(amr_sequences_test, sequence_type="amr")
    sentence_test = add_token(sentence_test, sequence_type="sentences")
    dataset_test = to_dataset(inpus_sequnce=amr_sequences_test, labels_sequence=sentence_test,tokenizer=tokenizer,
                         device=device)
    loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True)
    model.to(device)
    model.eval()
    decoded_sentences = []
    label_sentences = []
    with torch.no_grad():
        for batch in loader:
            generated_ids = model.generate(input_ids = batch["input_ids"], attention_mask = batch['attention_mask'],num_beams=5) # wenti chuxian
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            sentences_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
            decoded_sentences.append(sentences_decoded)
            label_sentences.append(decoded_labels)
    decoded_sentences = [item for sublist in decoded_sentences for item in sublist]
    label_sentences = [item for sublist in label_sentences for item in sublist]
    reference_sentences = []
    for label in label_sentences:
        reference_sentences.append([label])
    metric = load_metric("sacrebleu")
    results = metric.compute(predictions=decoded_sentences, references=reference_sentences)
    print(round(results["score"], 1))



