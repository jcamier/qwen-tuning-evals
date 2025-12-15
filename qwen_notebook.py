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
        json,
        np,
        os,
        rouge_scorer,
        sacrebleu,
        time,
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
    # print(chunks)

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

    # Explicitly avoid MPS to prevent compatibility issues
    # MPS has known issues with embedding layers and some operations
    if torch.cuda.is_available():
        device_name = "cuda"
        dtype = torch.float16
    else:
        # Always use CPU to avoid MPS issues (MPS has compatibility problems with some operations)
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

    mo.output.append(mo.md(f"""
    ✅ **Model loaded successfully!**

    - **Model**: {model_name}
    - **LoRA**: Enabled
    - **Device**: {device_name.upper()}
    - **Data Type**: {dtype}
    - **Parameters**: {sum(p.numel() for p in model.parameters()):,}"""))
    if device_name == "cpu":
        mo.output.append(mo.md("⚠️ **Note**: Using CPU for stability (MPS can have compatibility issues)"))
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
def _(device_name, json, mo, model, np, os, rouge_scorer, sacrebleu, tokenizer, torch, time):
    # Evaluation
    mo.md("### 📈 **Step 5: Comprehensive Evaluation**")

    # Ensure model is on the correct device (important after training)
    # Note: Can't reassign 'model' in marimo, so we call .to() which modifies in-place
    model.eval()

    # Disable MPS explicitly to prevent any accidental usage
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Ensure all model parameters are on the correct device
    # For PEFT models, this ensures both base model and adapters are moved
    # Calling without assignment works in-place for PyTorch models
    model.to(device_name)

    # Load evaluation dataset
    eval_data_path = "data/sample_evaluation_data.json"
    if os.path.exists(eval_data_path):
        # Use a unique file handle name to avoid redefining variables across cells in marimo
        with open(eval_data_path, 'r', encoding='utf-8') as eval_file:
            eval_data = json.load(eval_file)
        mo.md(f"✅ Loaded evaluation dataset: {len(eval_data)} samples")
    else:
        # Fallback to basic prompts if file doesn't exist
        eval_data = [
            {"prompt": "What is machine learning?", "reference": "Machine learning is a subset of AI.", "category": "definition"},
            {"prompt": "Explain AI.", "reference": "AI is artificial intelligence.", "category": "definition"}
        ]
        mo.md("⚠️ Using fallback evaluation prompts (evaluation data file not found)")

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    results = []
    generation_times = []

    with torch.no_grad():
        for item in eval_data:
            prompt = item["prompt"]
            reference = item.get("reference", "")

            # Format prompt
            formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

            # Tokenize and move to device
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(device_name) for k, v in inputs.items()}

            # Generate with timing
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_time = time.time() - start_time
            generation_times.append(gen_time)

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("assistant")[-1].strip()

            # Calculate metrics if reference is available
            rouge_scores = None
            bleu_score = None
            if reference:
                rouge_scores = rouge.score(reference, assistant_response)
                try:
                    bleu_score = sacrebleu.sentence_bleu(assistant_response, [reference]).score / 100.0
                except Exception:
                    bleu_score = 0.0

            results.append({
                "prompt": prompt,
                "response": assistant_response,
                "reference": reference,
                "category": item.get("category", "unknown"),
                "rouge1": rouge_scores['rouge1'].fmeasure if rouge_scores else None,
                "rouge2": rouge_scores['rouge2'].fmeasure if rouge_scores else None,
                "rougeL": rouge_scores['rougeL'].fmeasure if rouge_scores else None,
                "bleu": bleu_score if bleu_score is not None else None,
                "generation_time": gen_time,
                "response_length": len(assistant_response),
            })

    # Calculate aggregate metrics
    avg_length = np.mean([r["response_length"] for r in results])
    avg_time = np.mean(generation_times)
    tokens_per_second = np.mean([r["response_length"] / r["generation_time"] if r["generation_time"] > 0 else 0 for r in results])

    # Calculate average ROUGE and BLEU scores (only for items with references)
    rouge1_scores = [r["rouge1"] for r in results if r["rouge1"] is not None]
    rouge2_scores = [r["rouge2"] for r in results if r["rouge2"] is not None]
    rougeL_scores = [r["rougeL"] for r in results if r["rougeL"] is not None]
    bleu_scores = [r["bleu"] for r in results if r["bleu"] is not None]

    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else None
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else None
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else None
    avg_bleu = np.mean(bleu_scores) if bleu_scores else None

    # Display results
    metrics_section = ""
    if avg_rouge1 is not None:
        metrics_section = f"""
    📊 **Evaluation Metrics:**
    - **ROUGE-1**: {avg_rouge1:.4f} (higher is better, max 1.0)
    - **ROUGE-2**: {avg_rouge2:.4f} (higher is better, max 1.0)
    - **ROUGE-L**: {avg_rougeL:.4f} (higher is better, max 1.0)
    - **BLEU**: {avg_bleu:.4f} (higher is better, max 1.0)
    """

    performance_section = f"""
    ⚡ **Performance Metrics:**
    - **Average response length**: {avg_length:.0f} characters
    - **Average generation time**: {avg_time:.2f} seconds
    - **Average tokens/second**: {tokens_per_second:.1f}
    - **Total samples evaluated**: {len(results)}
    """

    # Sample responses
    sample_responses = "\n\n".join([
        f"""**Q{idx}** ({result['category']}): {result['prompt']}
**Response**: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}
{f"**ROUGE-1**: {result['rouge1']:.3f} | **BLEU**: {result['bleu']:.3f}" if result['rouge1'] is not None else ""}"""
        for idx, result in enumerate(results[:3], 1)
    ])

    mo.md(f"""
    ✅ **Evaluation completed!**

    {metrics_section}

    {performance_section}

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
