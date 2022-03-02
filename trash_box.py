# inputs = tokenizer(amr_sequences, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
# inputs['labels'] = inputs.input_ids.detach().clone()  # creat labels
# #create random array of floats with equal dimensions to input_ids tensor
# rand = torch.rand(inputs.input_ids.shape)
# # create mask array
# mask_arr = (rand < 0.15) * (inputs.input_ids != tokenizer.pad_token_id) * (
#             inputs.input_ids != tokenizer.bos_token_id) * (inputs.input_ids != tokenizer.eos_token_id)
#
# selection = []

# for i in range(inputs.input_ids.shape[0]):
#     selection.append(
#         torch.flatten(mask_arr[i].nonzero()).tolist()
#     )

# for i in range(inputs.input_ids.shape[0]):
#     inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id


#SBATCH --mem=64000
#SBATCH --gres=gpu
#SBATCH --ntasks=1


#loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,pin_memory=False)

# optim = AdamW(model.parameters(), lr=5e-5)


class AMRDataset(torch.utils.data.Dataset):  # turn them into dataset, but
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)



    # args = Seq2SeqTrainingArguments(
    #     output_dir='out_eva',
    #     overwrite_output_dir='True',
    #     save_strategy = "epoch",
    #     dataloader_pin_memory=False,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size = 1,
    #     num_train_epochs=1,
    #     dataloader_num_workers = 1,
    #     predict_with_generate=True
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=dataset,
    #     eval_dataset = dataset_test,
    # )
    predictions = trainer.predict(dataset_test)
    # gc.collect()
    # torch.cuda.empty_cache()
    # args = Seq2SeqTrainingArguments(
    #     output_dir='out_eva',
    #     overwrite_output_dir='True',
    #     save_strategy = "epoch",
    #     dataloader_pin_memory=False,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size = 1,
    #     num_train_epochs=1,
    #     predict_with_generate=True
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=dataset,
    #     eval_dataset = dataset_test,
    # # )
    # # predictions = trainer.predict(dataset_test)
    # metric = load_metric("sacrebleu")
    # metric.compute(predictions=predictions.predictions, references=predictions.label_ids)