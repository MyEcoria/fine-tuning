import transformers

# specify model configuration
model_name = "gpt2"
model_config = transformers.GPT2Config.from_pretrained(model_name)

# create the tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

# create the model
model = transformers.TFGPT2LMHeadModel.from_pretrained(model_name, config=model_config)

# specify training arguments
training_args = transformers.TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=1,
)

# create the text dataset
train_dataset = transformers.TextDataset(
    tokenizer=tokenizer,
    file_path="./data/train.txt",
    block_size=128,
)

# create the data collator
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# create the trainer and train the model
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# save the trained model
model.save_pretrained("./models/trained_model")
