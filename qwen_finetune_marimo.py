"""
Enhanced Qwen Fine-Tuning and Evaluation Framework using Marimo

This application provides a comprehensive interface for:
1. Fine-tuning Qwen models with LoRA
2. Comprehensive evaluation with multiple metrics
3. Visualization of training progress and results
4. Model comparison and analysis

Based on Eric Livesay's talk on Fine Tuning Qwen at UTA Python meetup.
"""

import marimo as mo
import os
import json
import time
import math
import textwrap
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
import sacrebleu
import nltk
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    model_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 2
    max_length: int = 1024
    output_dir: str = "./outputs"
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_4bit: bool = False
    use_8bit: bool = False

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True

@mo.app(title="Qwen Fine-Tune + Comprehensive Evals (Marimo)")
def app():
    """Main Marimo application for Qwen fine-tuning and evaluation"""

    # ========== UI COMPONENTS ==========

    # Model and Training Configuration
    model_config = mo.ui.form(
        mo.vstack([
            mo.md("### 🤖 Model Configuration"),
            mo.ui.text(
                label="Base Model (Hugging Face Hub ID)",
                value="Qwen/Qwen3-0.6B",
                full_width=True
            ),
            mo.ui.number(
                value=3,
                label="Epochs",
                min=1,
                max=10,
                step=1,
                full_width=True
            ),
            mo.ui.number(
                value=2e-5,
                label="Learning Rate",
                min=1e-6,
                max=1e-3,
                step=1e-6,
                full_width=True
            ),
            mo.ui.number(
                value=2,
                label="Batch Size",
                min=1,
                max=16,
                step=1,
                full_width=True
            ),
            mo.ui.number(
                value=1024,
                label="Max Sequence Length",
                min=256,
                max=4096,
                step=128,
                full_width=True
            ),
            mo.ui.text(
                label="Output Directory",
                value="./outputs",
                full_width=True
            ),
            mo.ui.number(
                value=42,
                label="Random Seed",
                min=0,
                max=10000,
                step=1,
                full_width=True
            ),
        ], gap="0.5em")
    )

    # LoRA Configuration
    lora_config = mo.ui.form(
        mo.vstack([
            mo.md("### 🔧 LoRA Configuration"),
            mo.ui.number(
                value=8,
                label="LoRA Rank (r)",
                min=1,
                max=64,
                step=1,
                full_width=True
            ),
            mo.ui.number(
                value=16,
                label="LoRA Alpha",
                min=1,
                max=128,
                step=1,
                full_width=True
            ),
            mo.ui.number(
                value=0.05,
                label="LoRA Dropout",
                min=0.0,
                max=0.5,
                step=0.01,
                full_width=True
            ),
            mo.ui.switch(
                label="Use 4-bit Quantization",
                value=False
            ),
            mo.ui.switch(
                label="Use 8-bit Quantization",
                value=False
            ),
        ], gap="0.5em")
    )

    # Data Configuration
    data_config = mo.ui.form(
        mo.vstack([
            mo.md("### 📊 Data Configuration"),
            mo.ui.dropdown(
                label="Data Source Type",
                options=["File", "URL", "HuggingFace Dataset"],
                value="File"
            ),
            mo.ui.text(
                label="Data Source Path/URL",
                value="data/training_data.txt",
                full_width=True
            ),
            mo.ui.number(
                value=2000,
                label="Chunk Size",
                min=500,
                max=4000,
                step=100,
                full_width=True
            ),
            mo.ui.number(
                value=200,
                label="Chunk Overlap",
                min=0,
                max=1000,
                step=50,
                full_width=True
            ),
            mo.ui.switch(
                label="Use Chat Template",
                value=True
            ),
        ], gap="0.5em")
    )

    # Evaluation Configuration
    eval_config = mo.ui.form(
        mo.vstack([
            mo.md("### 📈 Evaluation Configuration"),
            mo.ui.text_area(
                label="Evaluation Prompts (JSON format)",
                value=json.dumps([
                    "Summarize the key points of this text in 3 bullet points.",
                    "What are the main themes discussed?",
                    "Provide a brief analysis of the content.",
                    "What practical advice can be extracted?",
                    "Identify the most important concepts mentioned."
                ], indent=2),
                monospace=True,
                height="200px",
                full_width=True
            ),
            mo.ui.number(
                value=256,
                label="Max New Tokens",
                min=50,
                max=1024,
                step=50,
                full_width=True
            ),
            mo.ui.number(
                value=0.7,
                label="Temperature",
                min=0.1,
                max=2.0,
                step=0.1,
                full_width=True
            ),
            mo.ui.number(
                value=0.9,
                label="Top-p",
                min=0.1,
                max=1.0,
                step=0.05,
                full_width=True
            ),
            mo.ui.switch(
                label="Enable ROUGE Evaluation",
                value=True
            ),
            mo.ui.switch(
                label="Enable BLEU Evaluation",
                value=True
            ),
        ], gap="0.5em")
    )

    # Action Buttons
    action_buttons = mo.hstack([
        mo.ui.button(
            label="🚀 Start Training",
            kind="primary",
            disabled=False
        ),
        mo.ui.button(
            label="📊 Run Evaluations",
            kind="secondary",
            disabled=False
        ),
        mo.ui.button(
            label="💾 Save Model",
            kind="secondary",
            disabled=False
        ),
        mo.ui.button(
            label="📈 Generate Report",
            kind="secondary",
            disabled=False
        ),
    ], gap="1em")

    # ========== REACTIVE STATE ==========

    # Global state management
    app_state = mo.state({
        "phase": "idle",
        "log": [],
        "training_metrics": {"steps": [], "loss": [], "learning_rate": []},
        "evaluation_results": {},
        "model_loaded": False,
        "data_loaded": False,
        "training_complete": False
    })

    model_state = mo.state({
        "tokenizer": None,
        "model": None,
        "device": None,
        "config": None
    })

    data_state = mo.state({
        "dataset": None,
        "tokenized_dataset": None,
        "data_info": {}
    })

    # ========== UTILITY FUNCTIONS ==========

    def get_device_info() -> Tuple[str, bool]:
        """Get device information and capabilities"""
        use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            return "cuda", False
        elif use_mps:
            return "mps", True
        else:
            return "cpu", False

    def load_data_from_source(source_type: str, source_path: str) -> str:
        """Load data from various sources"""
        try:
            if source_type == "File":
                with open(source_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif source_type == "URL":
                import urllib.request
                with urllib.request.urlopen(source_path) as r:
                    return r.read().decode("utf-8", errors="ignore")
            elif source_type == "HuggingFace Dataset":
                # For HuggingFace datasets, we'll use a simple approach
                # In practice, you'd want to specify the dataset name and split
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                return "\n".join(dataset["text"][:1000])  # Limit for demo
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        i = 0
        while i < len(text):
            end_idx = min(i + chunk_size, len(text))
            chunk = text[i:end_idx].strip()
            if chunk:
                chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def format_chat_template(text: str, use_chat_template: bool = True) -> str:
        """Format text using Qwen's chat template"""
        if not use_chat_template:
            return text

        return f"""<|im_start|>system
You are a helpful assistant that provides accurate and informative responses.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []

        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })

        return {
            'rouge1': np.mean([s['rouge1'] for s in scores]),
            'rouge2': np.mean([s['rouge2'] for s in scores]),
            'rougeL': np.mean([s['rougeL'] for s in scores])
        }

    def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score"""
        # Tokenize predictions and references
        pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]

        return sacrebleu.corpus_bleu(pred_tokens, ref_tokens).score / 100

    # ========== DATA LOADING CELL ==========

    @mo.cell
    def data_loading_cell():
        """Load and preprocess training data"""
        try:
            # Get configuration values
            source_type = data_config.value.get("Data Source Type", "File")
            source_path = data_config.value.get("Data Source Path/URL", "data/training_data.txt")
            chunk_size = data_config.value.get("Chunk Size", 2000)
            overlap = data_config.value.get("Chunk Overlap", 200)
            use_chat_template = data_config.value.get("Use Chat Template", True)

            # Load data
            raw_text = load_data_from_source(source_type, source_path)
            chunks = chunk_text(raw_text, chunk_size, overlap)

            # Create dataset
            records = []
            for chunk in chunks:
                if chunk.strip():
                    formatted_text = format_chat_template(chunk, use_chat_template)
                    records.append({"text": formatted_text, "original": chunk})

            dataset = Dataset.from_list(records)

            # Update state
            data_state.value = {
                "dataset": dataset,
                "tokenized_dataset": None,
                "data_info": {
                    "total_chunks": len(dataset),
                    "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]),
                    "total_characters": len(raw_text)
                }
            }

            app_state.value["data_loaded"] = True
            app_state.value["log"].append(f"✅ Loaded {len(dataset)} training samples")

            return mo.md(f"""
            ### 📊 Data Loading Results

            - **Total chunks**: {len(dataset)}
            - **Average chunk length**: {data_state.value['data_info']['avg_chunk_length']:.0f} characters
            - **Total characters**: {data_state.value['data_info']['total_characters']:,}
            - **Chat template**: {'Enabled' if use_chat_template else 'Disabled'}
            """)

        except Exception as e:
            app_state.value["log"].append(f"❌ Data loading failed: {str(e)}")
            return mo.md(f"### ❌ Data Loading Error\n\n```\n{str(e)}\n```")

    # ========== MODEL LOADING CELL ==========

    @mo.cell
    def model_loading_cell():
        """Load and configure the model"""
        try:
            device, use_mps = get_device_info()

            # Get model configuration
            model_name = model_config.value.get("Base Model (Hugging Face Hub ID)", "Qwen/Qwen3-0.6B")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            # Configure quantization if needed
            quantization_config = None
            if lora_config.value.get("Use 4-bit Quantization", False):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            elif lora_config.value.get("Use 8-bit Quantization", False):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            elif device == "cuda":
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Configure LoRA
            lora_config_obj = LoraConfig(
                r=lora_config.value.get("LoRA Rank (r)", 8),
                lora_alpha=lora_config.value.get("LoRA Alpha", 16),
                lora_dropout=lora_config.value.get("LoRA Dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            # Prepare model for training
            if quantization_config:
                model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, lora_config_obj)

            # Update state
            model_state.value = {
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
                "config": lora_config_obj
            }

            app_state.value["model_loaded"] = True
            app_state.value["log"].append(f"✅ Model loaded: {model_name} on {device}")

            return mo.md(f"""
            ### 🤖 Model Configuration

            - **Model**: {model_name}
            - **Device**: {device} (MPS: {use_mps})
            - **LoRA Rank**: {lora_config_obj.r}
            - **LoRA Alpha**: {lora_config_obj.lora_alpha}
            - **LoRA Dropout**: {lora_config_obj.lora_dropout}
            - **Quantization**: {'4-bit' if lora_config.value.get('Use 4-bit Quantization') else '8-bit' if lora_config.value.get('Use 8-bit Quantization') else 'None'}
            """)

        except Exception as e:
            app_state.value["log"].append(f"❌ Model loading failed: {str(e)}")
            return mo.md(f"### ❌ Model Loading Error\n\n```\n{str(e)}\n```")

    # ========== TOKENIZATION CELL ==========

    @mo.cell
    def tokenization_cell():
        """Tokenize the dataset for training"""
        if not (model_state.value and data_state.value):
            return mo.md("⏳ Waiting for model and data to be loaded...")

        try:
            tokenizer = model_state.value["tokenizer"]
            dataset = data_state.value["dataset"]
            max_length = model_config.value.get("Max Sequence Length", 1024)

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )

            # Add labels for causal language modeling
            def add_labels(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples

            tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

            # Update state
            data_state.value["tokenized_dataset"] = tokenized_dataset

            app_state.value["log"].append(f"✅ Tokenized {len(tokenized_dataset)} samples")

            return mo.md(f"""
            ### 🔤 Tokenization Complete

            - **Tokenized samples**: {len(tokenized_dataset)}
            - **Max sequence length**: {max_length}
            - **Vocabulary size**: {tokenizer.vocab_size:,}
            """)

        except Exception as e:
            app_state.value["log"].append(f"❌ Tokenization failed: {str(e)}")
            return mo.md(f"### ❌ Tokenization Error\n\n```\n{str(e)}\n```")

    # ========== TRAINING CELL ==========

    @mo.cell
    def training_cell():
        """Execute the training process"""
        if not action_buttons.value.get("🚀 Start Training", False):
            return mo.md("⏳ Click 'Start Training' to begin...")

        if not (model_state.value and data_state.value.get("tokenized_dataset")):
            return mo.md("❌ Model and tokenized data required for training")

        try:
            # Set random seeds
            seed = model_config.value.get("Random Seed", 42)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Get training configuration
            epochs = model_config.value.get("Epochs", 3)
            learning_rate = model_config.value.get("Learning Rate", 2e-5)
            batch_size = model_config.value.get("Batch Size", 2)
            output_dir = model_config.value.get("Output Directory", "./outputs")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                logging_steps=5,
                save_strategy="epoch",
                evaluation_strategy="no",
                save_total_limit=2,
                load_best_model_at_end=False,
                report_to=None,  # Disable wandb for now
                remove_unused_columns=False,
                dataloader_drop_last=False,
                bf16=False,
                fp16=False,
                seed=seed,
            )

            # Custom callback for logging
            class TrainingCallback:
                def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                    if logs:
                        step = logs.get('step', 0)
                        loss = logs.get('loss', 0)
                        lr = logs.get('learning_rate', 0)

                        app_state.value["training_metrics"]["steps"].append(step)
                        app_state.value["training_metrics"]["loss"].append(loss)
                        app_state.value["training_metrics"]["learning_rate"].append(lr)

                        app_state.value["log"].append(f"Step {step}: Loss {loss:.4f}, LR {lr:.2e}")

            # Initialize trainer
            trainer = Trainer(
                model=model_state.value["model"],
                args=training_args,
                train_dataset=data_state.value["tokenized_dataset"],
                tokenizer=model_state.value["tokenizer"],
                callbacks=[TrainingCallback()],
            )

            # Start training
            app_state.value["phase"] = "training"
            app_state.value["log"].append("🚀 Starting training...")

            trainer.train()

            # Save model
            trainer.save_model()
            model_state.value["tokenizer"].save_pretrained(output_dir)

            app_state.value["phase"] = "completed"
            app_state.value["training_complete"] = True
            app_state.value["log"].append("✅ Training completed successfully!")

            return mo.md(f"""
            ### 🎉 Training Complete!

            - **Output directory**: {output_dir}
            - **Final loss**: {app_state.value['training_metrics']['loss'][-1]:.4f}
            - **Total steps**: {len(app_state.value['training_metrics']['steps'])}
            """)

        except Exception as e:
            app_state.value["phase"] = "error"
            app_state.value["log"].append(f"❌ Training failed: {str(e)}")
            return mo.md(f"### ❌ Training Error\n\n```\n{str(e)}\n```")

    # ========== TRAINING VISUALIZATION CELL ==========

    @mo.cell
    def training_visualization_cell():
        """Display training progress and metrics"""
        if not app_state.value["training_metrics"]["steps"]:
            return mo.md("📊 Training metrics will appear here during training...")

        try:
            steps = app_state.value["training_metrics"]["steps"]
            loss = app_state.value["training_metrics"]["loss"]
            lr = app_state.value["training_metrics"]["learning_rate"]

            # Create subplots
            fig = go.Figure()

            # Loss plot
            fig.add_trace(go.Scatter(
                x=steps,
                y=loss,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title="Training Loss Over Time",
                xaxis_title="Training Steps",
                yaxis_title="Loss",
                hovermode='x unified',
                template="plotly_white"
            )

            return mo.ui.plotly(fig)

        except Exception as e:
            return mo.md(f"❌ Visualization error: {str(e)}")

    # ========== EVALUATION CELL ==========

    @mo.cell
    def evaluation_cell():
        """Run comprehensive evaluations"""
        if not action_buttons.value.get("📊 Run Evaluations", False):
            return mo.md("⏳ Click 'Run Evaluations' to start...")

        if not (model_state.value and app_state.value["training_complete"]):
            return mo.md("❌ Model must be trained before evaluation")

        try:
            # Get evaluation configuration
            eval_prompts = json.loads(eval_config.value.get("Evaluation Prompts (JSON format)", "[]"))
            max_new_tokens = eval_config.value.get("Max New Tokens", 256)
            temperature = eval_config.value.get("Temperature", 0.7)
            top_p = eval_config.value.get("Top-p", 0.9)
            enable_rouge = eval_config.value.get("Enable ROUGE Evaluation", True)
            enable_bleu = eval_config.value.get("Enable BLEU Evaluation", True)

            model = model_state.value["model"]
            tokenizer = model_state.value["tokenizer"]
            device = model_state.value["device"]

            model.eval()
            results = []

            app_state.value["log"].append("📊 Starting evaluation...")

            with torch.no_grad():
                for i, prompt in enumerate(tqdm(eval_prompts, desc="Evaluating")):
                    # Format prompt
                    formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

                    # Tokenize
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    assistant_response = response.split("assistant")[-1].strip()

                    results.append({
                        "prompt": prompt,
                        "response": assistant_response,
                        "full_response": response
                    })

            # Calculate metrics if enabled
            metrics = {}
            if enable_rouge and len(results) > 1:
                # Use first response as reference for others (simplified)
                references = [results[0]["response"]] * len(results)
                predictions = [r["response"] for r in results]
                metrics["rouge"] = calculate_rouge_scores(predictions, references)

            if enable_bleu and len(results) > 1:
                references = [results[0]["response"]] * len(results)
                predictions = [r["response"] for r in results]
                metrics["bleu"] = calculate_bleu_score(predictions, references)

            # Update state
            app_state.value["evaluation_results"] = {
                "results": results,
                "metrics": metrics,
                "config": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }

            app_state.value["log"].append("✅ Evaluation completed!")

            # Create results table
            df = pd.DataFrame(results)

            return mo.ui.table(
                df[["prompt", "response"]],
                page_size=10,
                selection=None
            )

        except Exception as e:
            app_state.value["log"].append(f"❌ Evaluation failed: {str(e)}")
            return mo.md(f"### ❌ Evaluation Error\n\n```\n{str(e)}\n```")

    # ========== METRICS VISUALIZATION CELL ==========

    @mo.cell
    def metrics_visualization_cell():
        """Display evaluation metrics"""
        if not app_state.value["evaluation_results"]:
            return mo.md("📈 Evaluation metrics will appear here after running evaluations...")

        try:
            metrics = app_state.value["evaluation_results"]["metrics"]

            if not metrics:
                return mo.md("No metrics available. Enable ROUGE or BLEU evaluation.")

            # Create metrics visualization
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            fig = go.Figure(data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                )
            ])

            fig.update_layout(
                title="Evaluation Metrics",
                xaxis_title="Metric",
                yaxis_title="Score",
                template="plotly_white"
            )

            return mo.ui.plotly(fig)

        except Exception as e:
            return mo.md(f"❌ Metrics visualization error: {str(e)}")

    # ========== STATUS LOG CELL ==========

    @mo.cell
    def status_log_cell():
        """Display application status and logs"""
        logs = app_state.value["log"][-20:]  # Show last 20 logs
        log_text = "\n".join(logs) if logs else "No logs yet..."

        return mo.ui.code(
            log_text,
            language="text",
            height="200px"
        )

    # ========== MAIN LAYOUT ==========

    return mo.vstack([
        mo.md("# 🚀 Qwen Fine-Tuning & Evaluation Framework"),
        mo.md("Based on Eric Livesay's talk on Fine Tuning Qwen at UTA Python meetup"),

        mo.hstack([
            model_config,
            lora_config
        ], gap="2em"),

        mo.hstack([
            data_config,
            eval_config
        ], gap="2em"),

        action_buttons,

        mo.md("### 📊 Status & Logs"),
        status_log_cell,

        mo.md("### 📈 Data Loading"),
        data_loading_cell,

        mo.md("### 🤖 Model Loading"),
        model_loading_cell,

        mo.md("### 🔤 Tokenization"),
        tokenization_cell,

        mo.md("### 🏋️ Training"),
        training_cell,

        mo.md("### 📊 Training Progress"),
        training_visualization_cell,

        mo.md("### 📈 Evaluation"),
        evaluation_cell,

        mo.md("### 📊 Evaluation Metrics"),
        metrics_visualization_cell,

    ], gap="1.5em")

if __name__ == "__main__":
    mo.main()
