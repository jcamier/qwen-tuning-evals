import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.output.append(mo.md("# 🚀 Qwen Fine-Tuning & Evaluation Framework"))
    mo.output.append(mo.md("Based on Eric Livesay's talk on Fine Tuning Qwen at UTA Python meetup"))
    mo.output.append(mo.md("**Run each cell below in order to fine-tune your Qwen model!**"))
    return


@app.cell
def _(mo):
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

    # Set environment variables to suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    mo.output.append(mo.md("✅ **All libraries imported successfully!**"))

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Dataset,
        LoraConfig,
        TaskType,
        Trainer,
        TrainingArguments,
        get_peft_model,
        np,
        os,
        prepare_model_for_kbit_training,
        rouge_scorer,
        torch,
    )


@app.cell
def _(mo):
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
    return (
        batch_size,
        chunk_size,
        data_file,
        epochs,
        learning_rate,
        model_name,
        output_dir,
    )


@app.cell
def _(Dataset, chunk_size, data_file, mo):
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
    return (dataset,)


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    LoraConfig,
    TaskType,
    get_peft_model,
    mo,
    model_name,
    os,
    torch,
):
    # Load model and tokenizer
    mo.md("### 🤖 **Step 2: Loading Model**")
    mo.md(f"⏳ Loading {model_name}...")

    # Determine device - force CPU to avoid MPS issues on Mac
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for MPS issues

    if torch.cuda.is_available():
        device_name = "cuda"
        dtype = torch.float16
    else:
        # Use CPU to avoid MPS issues
        device_name = "cpu"
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Move to device
    model = model.to(device_name)

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

    mo.md(f"""
    ✅ **Model loaded successfully!**

    - **Model**: {model_name}
    - **LoRA**: Enabled
    - **Device**: {device_name.upper()}
    - **Data Type**: {dtype}
    - **Parameters**: {sum(p.numel() for p in model.parameters()):,}

    ⚠️ **Note**: Using CPU for stability (MPS can have compatibility issues)
    """)
    return device_name, model, tokenizer


@app.cell
def _(dataset, mo, tokenizer):
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
    return (tokenized_dataset,)


@app.cell
def _(
    Trainer,
    TrainingArguments,
    batch_size,
    epochs,
    learning_rate,
    mo,
    model,
    output_dir,
    tokenized_dataset,
    tokenizer,
):
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
        processing_class=tokenizer,
    )

    mo.md("🚀 **Starting training...**")

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Get the final loss (search backwards for the last entry with 'loss')
    final_loss = None
    for log_entry in reversed(trainer.state.log_history):
        if 'loss' in log_entry:
            final_loss = log_entry['loss']
            break

    loss_text = f"{final_loss:.4f}" if final_loss is not None else "N/A"

    mo.md(f"""
    ✅ **Training completed successfully!**

    - **Model saved to**: {output_dir}
    - **Epochs**: {epochs}
    - **Final loss**: {loss_text}
    - **Total training steps**: {trainer.state.global_step}
    """)
    return


@app.cell
def _(device_name, mo, model, np, tokenizer, torch):
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

            # Tokenize and move to device
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(device_name) for k, v in inputs.items()}

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
    return


@app.cell
def _(mo):
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
    return


if __name__ == "__main__":
    app.run()
