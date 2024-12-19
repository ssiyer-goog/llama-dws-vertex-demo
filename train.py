import os
import argparse
from google.cloud import aiplatform
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import transformers
from peft import LoraConfig, get_peft_model

# Ensure transformers version is compatible
assert transformers.__version__ >= "4.34.1"

# Define the model ID and dataset name
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME = "rotten_tomatoes"

def main(args):
    # Initialize Vertex AI
    aiplatform.init(project=args.project_id, location=args.location)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def preprocess_data(tokenizer, examples):
    inputs = [f"Classify the sentiment of this text: {text}" for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # Ensure labels are in the correct format
    if "label" in examples:
        model_inputs["labels"] = torch.tensor(examples["label"])
    else:
        raise ValueError("The dataset does not contain a 'label' column.")

    return model_inputs


    # Load and preprocess the dataset
    dataset = load_dataset(DATASET_NAME)
    tokenized_datasets = dataset.map(lambda examples: preprocess_data(tokenizer, examples), batched=True, num_proc=4, remove_columns=dataset["train"].column_names)

    # Add debugging prints here
    print("Train dataset length:", len(tokenized_datasets["train"]))
    print("Validation dataset length:", len(tokenized_datasets["validation"]))
    print("Sample train input:", tokenized_datasets["train"][0])
    print("Sample train input keys:", tokenized_datasets["train"][0].keys())
    
    # Convert the dataset to PyTorch format
    tokenized_datasets.set_format("torch")

    # Add this block
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    # Create a BitsAndBytesConfig for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True
    )

    # Load the model in 8-bit with BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=args.hf_token,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Configure LoRA
    model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )

    # Get PEFT model with LoRA adapter
    model = get_peft_model(model, peft_config)

    # Print details of trainable parameters
    model.print_trainable_parameters()

    # enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=f'{args.output_dir}/logs',
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

   
    # Then update your Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,  # Add this line
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True, help='Your Google Cloud project ID')
    parser.add_argument('--location', type=str, required=True, help='Your Google Cloud location')
    parser.add_argument('--hf_token', type=str, required=True, help='Your Hugging Face access token')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for the model')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    args = parser.parse_args()
    main(args)
    