from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments
from transformers import Trainer
import random
import os
from datasets import load_metric
from ultils import *
from tqdm import tqdm
import penman

def masking_rate_calulation(step,length):
    masking_rate = 0.3+step/length*0.8
    if masking_rate<1:
        return masking_rate
    else:
        return 1

def mask_sentence_unified(tokenizer,sentences,masking_rate,device):
    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding='max_length',add_special_tokens = False)
    inputs['labels'] = inputs.input_ids.detach().clone()  # creat labels
    rand = torch.rand(inputs.input_ids.shape)  # random number
    mask_arr = (rand < masking_rate) * (inputs.input_ids != tokenizer.pad_token_id) * (
                 inputs.input_ids != tokenizer.bos_token_id) * (inputs.input_ids != tokenizer.eos_token_id)
    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"]=inputs["attention_mask"].to(device)
    inputs["labels"] = inputs["labels"].to(device)
    decoded_inputs = tokenizer.batch_decode(inputs["input_ids"], clean_up_tokenization_spaces=False)
    #print(decoded_inputs)
    return decoded_inputs

def unified_training(amr_squences,sentences,masking_rate,tokenizer,device):
    masked_amr = masking_amr(amr_list = amr_squences, masking_rate = masking_rate)
    masked_amr = add_token(sequence_list = masked_amr,sequence_type = "amr")
    masked_sentence = mask_sentence_unified(tokenizer = tokenizer,sentences = sentences,masking_rate = masking_rate,device = device)
    masked_sentence = add_token(sequence_list = masked_amr,sequence_type = "sentence")
    ziped_pairs = zip(masked_amr,masked_sentence)
    new_pairs = []
    for (amr,sentence) in ziped_pairs:
        new_data = amr+" "+sentence
        new_pairs.append(new_data)
    return new_pairs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
     # Get the device
    sentences,amr_sequences = open_dataset(path = "/home/students/he/pythonProject1/training")
    tokenizer = BartTokenizer.from_pretrained('/home/students/he/pythonProject1/bart_model_sentence',local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained('/home/students/he/pythonProject1/bart_model_sentence',local_files_only=True)
    ziped_pairs = list(zip(sentences, amr_sequences))
    loader = torch.utils.data.DataLoader(ziped_pairs, batch_size=4, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)
    steps = len(loader)
    epochs = 1
    model.to(device)
    model.train()
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for step,batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            sentences = batch[0] # unzip the batch unzip the batch
            amr = batch[1]
            mask_rate =  masking_rate_calulation(step = step,length = len(loader))
            masked_pairs = unified_training(amr_squences = amr,sentences = sentences,masking_rate = mask_rate ,tokenizer = tokenizer,device = device)
            encoded_new_pairs = tokenizer(masked_pairs, return_tensors='pt', max_length=512, truncation=True,
                                          padding='max_length', additional_special_tokens=False).to(device)
            labels = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True,
                               padding='max_length').to(device)

            encoded_new_pairs["labels"] = labels["input_ids"].to(device)
            encoded_new_pairs["input_ids"] =  encoded_new_pairs["input_ids"].to(device)
            encoded_new_pairs["attention_mask"] = encoded_new_pairs["attention_mask"].to(device)
            encoded_new_pairs["labels"] = encoded_new_pairs["labels"].to(device)
            # process
            outputs = model(**encoded_new_pairs)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())





