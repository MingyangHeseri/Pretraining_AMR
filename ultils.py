from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from transformers import AdamW
from transformers import TrainingArguments
from transformers import Trainer
import random
import os
from datasets import load_metric
import numpy as np
import penman

def open_amr(dataname):
  with open(dataname,"r") as file:
    text = file.read()
  amrs = text.split("\n\n")
  sentences = []
  amr_sequences = []
  amrs = [amr for amr in amrs if amr]
  for amr in amrs:
    amr = amr.split("\n")
    id = amr[0]
    sent = amr[1].replace("# ::snt ", "").replace("# ::tok ", "").strip().lower()
    sentences.append(sent)
    concepts = amr[3:]
    decoded_amr = penman.decode(amr)
    variables = [variable for variable,_,_ in decoded_amr.instances()]
    amr_sequence = []
    variable_flag = 1
    for concept in concepts:
      concept_elements = concept.strip().split()
      preprocessed_element = []
      for element in concept_elements:
        if "(" in element:
          element = element.replace("(", "( ")
        if ")" in element:
          element = element.replace(")", " )")
        if element == "/":
          continue
        if any(item in variables for item in element.split()):
          for item in element.split():
            if item in variables:
              element_variable = item
          element = element.replace(element_variable,"<"+"r"+str(variable_flag)+">")
          variable_flag = variable_flag+1
        preprocessed_element.append(element)
      preprocessed_element = " ".join(preprocessed_element)
      amr_sequence.append(preprocessed_element)
    amr_sequence = " ".join(amr_sequence)
    amr_sequences.append(amr_sequence)
  return sentences,amr_sequences

# def open_amr(dataname):
#     with open(dataname, "r") as file:
#         text = file.read()
#     amrs = text.split("\n\n")
#     sentences = []
#     amr_sequences = []
#     amrs = [amr for amr in amrs if amr]
#     for amr in amrs:
#         amr = amr.split("\n")
#         id = amr[0]
#         sent = amr[1].replace("# ::snt ", "").replace("# ::tok ", "").strip().lower()
#         sentences.append(sent)
#         concepts = amr[3:]
#         amr_sequence = []
#         for concept in concepts:
#             concept_elements = concept.strip().split()
#             preprocessed_element = []
#             for element in concept_elements:
#                 if "(" in element:
#                     element = element.replace("(", "( ")
#                 if ")" in element:
#                     element = element.replace(")", " )")
#                 preprocessed_element.append(element)
#             preprocessed_element = " ".join(preprocessed_element)
#             amr_sequence.append(preprocessed_element)
#         amr_sequence = " ".join(amr_sequence)
#         amr_sequences.append(amr_sequence)
#     return sentences, amr_sequences


def masking_amr(amr_list,masking_rate):
    # read in the amr and make it a list in list, in which the [0] represent the input_id and the
    # mask the node and the edge with
    new_amr_list_masked = []
    for amr in amr_list:
        new_amr = amr.split()
        masked_amr = []
        for token in new_amr:
            if token.startswith(":"):  # mask the edge
                if random.random() < masking_rate:
                    token = "<mask>"
                    masked_amr.append(token)
                else:
                    masked_amr.append(token)
            elif token not in "()" and len(token) != 1:  # mask the concept
                if random.random() < masking_rate:
                    token = "<mask>"
                    masked_amr.append(token)
                else:
                    masked_amr.append(token)
            else:
                masked_amr.append(token)
        masked_amr = " ".join(masked_amr)
        new_amr_list_masked.append(masked_amr)
    return new_amr_list_masked


def mask_subgraph(amr_list,masking_rate):
  sub_graphs_to_mask = {}
  for amr in amr_list:
    sub_graphs_to_mask[amr] = []
    new_amr = amr.split()
    masked_amr = []
    for idx,token in enumerate(new_amr):
      subgraph = []
      try:
        if token.startswith(":") and new_amr[idx+1] is "(" :  # mask the subgraph      # mask the subgraph
          if random.random() <masking_rate:
             subgraph.append(token)
             sub_in_sub = 0
             sub_amr_list = new_amr[idx+1:]   # try to get the whole subgraph
             for idx1,tok in enumerate(sub_amr_list):
               if tok is not ")":
                 if not tok.startswith(":"):
                   subgraph.append(tok)
                 else:
                   try:
                     if sub_amr_list[idx1+1] is "(":
                       sub_in_sub+=1
                   except IndexError:
                     pass
                   subgraph.append(tok)
               else:
                 if sub_in_sub != 0:    #if it is not at the end of this subgraph
                   subgraph.append(tok)
                   sub_in_sub -=1
                 else:
                    subgraph.append(tok)           # reach the end of this subgraph
                    break
      except IndexError:
        pass
      if len(subgraph) != 0:
        subgraph = " ".join(subgraph)
        sub_graphs_to_mask[amr].append(subgraph)
  new_amr_list_masked = {}
  for key, value in sub_graphs_to_mask.items():
      new_list = []
      for subgraph in value:
          for subgraph1 in value:
              if subgraph1 in subgraph and subgraph1 != subgraph:  # if it is smaller than a subgraph to be masked, then there is no need to keep it
                  pass
              else:
                  new_list.append(subgraph)
      new_amr_list_masked[key] = new_list
  amr_masked = []
  for key, value in new_amr_list_masked.items():
      for amr_mask in value:
          key = key.replace(amr_mask, "<mask>")
      amr_masked.append(key)
  return amr_masked



def open_dataset(path):
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    sentences = []
    amr_sequences = []
    for text in files_txt:
        path_text = path + "/" + text
        sentence, amr_sequence =open_amr(path_text)
        sentences.append(sentence)
        amr_sequences.append(amr_sequence)
    sentences = [item for sublist in sentences for item in sublist]
    amr_sequences = [item for sublist in amr_sequences for item in sublist]
    return sentences, amr_sequences


class AMRDataset(torch.utils.data.Dataset):  # turn them into dataset
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def to_dataset(inpus_sequnce,labels_sequence,tokenizer,device):
    inputs = tokenizer(inpus_sequnce, return_tensors='pt', max_length=512, truncation=True, padding='max_length',add_special_tokens = False).to(device)
    labels = tokenizer(labels_sequence, return_tensors='pt', max_length=512, truncation=True, padding='max_length',add_special_tokens = False).to(device)
    inputs["labels"] = labels["input_ids"].to(device)
    print(inputs)
    dataset = AMRDataset(inputs)
    return dataset


def mask_sentence(tokenizer,sentences,masking_rate,device):
    inputs = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
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
    dataset = AMRDataset(inputs)
    return dataset

def add_token(sequence_list,sequence_type):
    if sequence_type is "amr":
        start_token = "<g>"
        end_token = "</g>"
    else:
        start_token = "<s>"
        end_token = "</s>"
    sequence_list_with_token = []
    for i in sequence_list:
        i = start_token+" "+i+" "+end_token
        sequence_list_with_token.append(i)
    return sequence_list_with_token

def compute_metrics(eval_preds):
    metric = load_metric("bleu")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)