import marimo as mo

__generated_with = "0.17.0"
app = mo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("# 🚀 Qwen Fine-Tuning & Evaluation Framework")
    mo.md("Based on Eric Livesay's talk on Fine Tuning Qwen at UTA Python meetup")
    mo.md("**Run each cell below in order to fine-tune your Qwen model!**")


@app.cell
def __(mo):
    import os
    import json
    import time
    import torch
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from rouge_score import rouge_scorer
    import sacrebleu
    import nltk

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    mo.md("✅ **All libraries imported successfully!**")
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Dataset,
        LoraConfig,
        TaskType,
        Trainer,
        TrainingArguments,
        get_peft_model,
        json,
        mo,
        nltk,
        np,
        os,
        pd,
        prepare_model_for_kbit_training,
        rouge_scorer,
        sacrebleu,
        time,
        torch,
    )


@app.cell
def __(mo):
    # Configuration
    model_name = "Qwen/Qwen3-0.6B"
    data_file = "data/sample_training_data.txt"
    output_dir = "./outputs"
    epochs = 1  # Reduced for demo
    learning_rate = 2e-5
    batch_size = 1  # Reduced for demo
    chunk_size = 500  # Reduced for demo

    mo.md(f"""
    ### 📋 **Configuration**
    - **Model**: {model_name}
    - **Data**: {data_file}
    - **Output**: {output_dir}
    - **Epochs**: {epochs}
    - **Learning Rate**: {learning_rate}
    - **Batch Size**: {batch_size}
    - **Chunk Size**: {chunk_size}
    """)


@app.cell
def __(data_file, mo, chunk_size):
    # Load and process data
    mo.md("### 📊 **Step 1: Loading Data**")

    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Simple chunking - using list comprehension to avoid variable conflicts
    chunks = [
        text[start:start+chunk_size].strip()
        for start in range(0, len(text), chunk_size)
        if text[start:start+chunk_size].strip()
    ]

    # Format with chat template - using list comprehension
    formatted_chunks = [
        f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{chunk}<|im_end|>
<|im_start|>assistant
"""
        for chunk in chunks
    ]

    # Create dataset
    dataset = Dataset.from_list([{"text": chunk} for chunk in formatted_chunks])

    mo.md(f"""
    ✅ **Data loaded successfully!**

    - **File**: {data_file}
    - **Characters**: {len(text):,}
    - **Chunks**: {len(chunks)}
    - **Dataset size**: {len(dataset)}
    """)


@app.cell
def __(mo, model_name):
    # Load model and tokenizer
    mo.md("### 🤖 **Step 2: Loading Model**")
    mo.md(f"⏳ Loading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    mo.md(f"""
    ✅ **Model loaded successfully!**

    - **Model**: {model_name}
    - **LoRA**: Enabled
    - **Device**: {device}
    - **Parameters**: {sum(p.numel() for p in model.parameters()):,}
    """)


@app.cell
def __(dataset, mo, tokenizer):
    # Tokenize dataset
    mo.md("### 🔤 **Step 3: Tokenizing Dataset**")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,  # Reduced for demo
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Add labels
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

    mo.md(f"""
    ✅ **Dataset tokenized successfully!**

    - **Tokenized samples**: {len(tokenized_dataset)}
    - **Max length**: 512 tokens
    - **Vocabulary size**: {tokenizer.vocab_size:,}
    """)


@app.cell
def __(batch_size, epochs, learning_rate, mo, output_dir, tokenizer, model, tokenized_dataset):
    # Training
    mo.md("### 🏋️ **Step 4: Training**")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=None,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    mo.md("🚀 **Starting training...**")

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    mo.md(f"""
    ✅ **Training completed successfully!**

    - **Model saved to**: {output_dir}
    - **Epochs**: {epochs}
    - **Final loss**: {trainer.state.log_history[-1]['loss']:.4f}
    """)


@app.cell
def __(mo, model, tokenizer):
    # Evaluation
    mo.md("### 📈 **Step 5: Evaluation**")

    # Sample evaluation prompts
    eval_prompts = [
        "What is machine learning?",
        "Explain the concept of artificial intelligence.",
        "What are the benefits of renewable energy?"
    ]

    model.eval()
    results = []

    with torch.no_grad():
        for prompt in eval_prompts:
            # Format prompt
            formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt")

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("assistant")[-1].strip()

            results.append({
                "prompt": prompt,
                "response": assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
            })

    # Calculate simple metrics
    avg_length = np.mean([len(r["response"]) for r in results])

    # Display results with sample Q&A
    sample_responses = "\n\n".join([
        f"**Q{idx}**: {result['prompt']}\n**A{idx}**: {result['response']}"
        for idx, result in enumerate(results[:2], 1)
    ])

    mo.md(f"""
    ✅ **Evaluation completed!**

    📊 **Results:**
    - **Average response length**: {avg_length:.0f} characters
    - **Responses generated**: {len(results)}

    📝 **Sample responses:**

    {sample_responses}
    """)


@app.cell
def __(mo):
    mo.md("### 🎉 **Summary**")
    mo.md("""
    **Congratulations!** You have successfully:

    1. ✅ Loaded and processed training data
    2. ✅ Loaded Qwen3-0.6B model with LoRA
    3. ✅ Tokenized the dataset
    4. ✅ Fine-tuned the model
    5. ✅ Evaluated the model performance

    **Your fine-tuned model is saved in the `./outputs` directory!**
    """)


if __name__ == "__main__":
    app.run()
