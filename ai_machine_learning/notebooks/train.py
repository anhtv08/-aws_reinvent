
import argparse
import os
import pandas as pd
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Dataset
import torch

def model_fn(model_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(os.path.join(args.train, "summarization_train.csv"))
    dataset = Dataset.from_pandas(df)
    
    # Load tokenizer and model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    prefix = "summarize: "
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
