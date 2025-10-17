"""
Comprehensive Evaluation Framework for Fine-tuned Qwen Models

This module provides advanced evaluation capabilities including:
- Multiple evaluation metrics (ROUGE, BLEU, BERTScore, etc.)
- Automated evaluation pipelines
- Model comparison utilities
- Evaluation report generation
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Evaluation metrics
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr

# Download required NLTK data
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt')

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

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
    batch_size: int = 1
    device: str = "auto"

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bleu: float = 0.0
    bert_score_f1: float = 0.0
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    exact_match: float = 0.0
    semantic_similarity: float = 0.0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0

class QwenEvaluator:
    """Comprehensive evaluator for fine-tuned Qwen models"""

    def __init__(
        self,
        model_path: str,
        config: Optional[EvaluationConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the evaluator

        Args:
            model_path: Path to the fine-tuned model
            config: Evaluation configuration
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.config = config or EvaluationConfig()
        self.device = device or self._get_device()

        # Load model and tokenizer
        self._load_model()

        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        logger.info(f"Evaluator initialized with model: {model_path}")
        logger.info(f"Device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True
            )

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        config: Optional[EvaluationConfig] = None
    ) -> Tuple[str, float]:
        """
        Generate a response for a given prompt

        Args:
            prompt: Input prompt
            config: Optional evaluation configuration

        Returns:
            Tuple of (generated_text, generation_time)
        """
        config = config or self.config

        # Format prompt with chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generation_time = time.time() - start_time

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "assistant" in full_response:
            response = full_response.split("assistant")[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()

        return response, generation_time

    def evaluate_single(
        self,
        prompt: str,
        reference: str,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a single prompt-reference pair

        Args:
            prompt: Input prompt
            reference: Reference answer
            config: Optional evaluation configuration

        Returns:
            EvaluationMetrics object
        """
        config = config or self.config

        # Generate response
        response, generation_time = self.generate_response(prompt, config)

        # Calculate metrics
        metrics = EvaluationMetrics()
        metrics.generation_time = generation_time

        # Calculate tokens per second
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(response))
        metrics.tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, response)
        metrics.rouge_1 = rouge_scores['rouge1'].fmeasure
        metrics.rouge_2 = rouge_scores['rouge2'].fmeasure
        metrics.rouge_l = rouge_scores['rougeL'].fmeasure

        # BLEU score
        try:
            metrics.bleu = sacrebleu.sentence_bleu(response, [reference]).score / 100
        except:
            metrics.bleu = 0.0

        # Exact match
        metrics.exact_match = 1.0 if response.strip().lower() == reference.strip().lower() else 0.0

        # BERTScore (optional, can be slow)
        try:
            P, R, F1 = bert_score([response], [reference], lang="en", verbose=False)
            metrics.bert_score_precision = P.item()
            metrics.bert_score_recall = R.item()
            metrics.bert_score_f1 = F1.item()
        except:
            metrics.bert_score_precision = 0.0
            metrics.bert_score_recall = 0.0
            metrics.bert_score_f1 = 0.0

        return metrics

    def evaluate_batch(
        self,
        prompts: List[str],
        references: List[str],
        config: Optional[EvaluationConfig] = None
    ) -> List[EvaluationMetrics]:
        """
        Evaluate a batch of prompt-reference pairs

        Args:
            prompts: List of input prompts
            references: List of reference answers
            config: Optional evaluation configuration

        Returns:
            List of EvaluationMetrics objects
        """
        if len(prompts) != len(references):
            raise ValueError("Number of prompts must match number of references")

        results = []
        for prompt, reference in zip(prompts, references):
            try:
                metrics = self.evaluate_single(prompt, reference, config)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating prompt: {e}")
                # Add empty metrics for failed evaluation
                results.append(EvaluationMetrics())

        return results

    def evaluate_dataset(
        self,
        dataset: Union[Dataset, List[Dict[str, str]]],
        config: Optional[EvaluationConfig] = None
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset

        Args:
            dataset: Dataset with 'prompt' and 'reference' columns
            config: Optional evaluation configuration

        Returns:
            Dictionary with evaluation results
        """
        if isinstance(dataset, Dataset):
            prompts = dataset['prompt']
            references = dataset['reference']
        else:
            prompts = [item['prompt'] for item in dataset]
            references = [item['reference'] for item in dataset]

        # Run evaluation
        results = self.evaluate_batch(prompts, references, config)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(results)

        # Add individual results
        aggregated['individual_results'] = [asdict(metric) for metric in results]

        return aggregated

    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across multiple evaluations"""
        if not metrics_list:
            return {}

        # Convert to arrays for easier computation
        arrays = {}
        for field in EvaluationMetrics.__dataclass_fields__:
            arrays[field] = [getattr(metric, field) for metric in metrics_list]

        aggregated = {}
        for field, values in arrays.items():
            if values and any(v is not None for v in values):
                aggregated[f"{field}_mean"] = np.mean(values)
                aggregated[f"{field}_std"] = np.std(values)
                aggregated[f"{field}_min"] = np.min(values)
                aggregated[f"{field}_max"] = np.max(values)
                aggregated[f"{field}_median"] = np.median(values)

        # Add summary statistics
        aggregated['total_samples'] = len(metrics_list)
        aggregated['successful_evaluations'] = sum(1 for m in metrics_list if m.generation_time > 0)

        return aggregated

    def compare_models(
        self,
        other_model_path: str,
        prompts: List[str],
        references: List[str],
        config: Optional[EvaluationConfig] = None
    ) -> Dict[str, Any]:
        """
        Compare this model with another model

        Args:
            other_model_path: Path to the other model
            prompts: List of prompts
            references: List of references
            config: Optional evaluation configuration

        Returns:
            Comparison results
        """
        # Evaluate current model
        current_results = self.evaluate_batch(prompts, references, config)
        current_aggregated = self._aggregate_metrics(current_results)

        # Load and evaluate other model
        other_evaluator = QwenEvaluator(other_model_path, config, self.device)
        other_results = other_evaluator.evaluate_batch(prompts, references, config)
        other_aggregated = self._aggregate_metrics(other_results)

        # Compare metrics
        comparison = {}
        for metric in ['rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'bert_score_f1']:
            current_mean = current_aggregated.get(f"{metric}_mean", 0)
            other_mean = other_aggregated.get(f"{metric}_mean", 0)

            comparison[f"{metric}_current"] = current_mean
            comparison[f"{metric}_other"] = other_mean
            comparison[f"{metric}_improvement"] = current_mean - other_mean
            comparison[f"{metric}_improvement_pct"] = (
                (current_mean - other_mean) / other_mean * 100
                if other_mean > 0 else 0
            )

        comparison['current_model'] = self.model_path
        comparison['other_model'] = other_model_path

        return comparison

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            evaluation_results: Results from evaluate_dataset
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = []
        report.append("# Qwen Model Evaluation Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_path}")
        report.append("")

        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"Total samples: {evaluation_results.get('total_samples', 0)}")
        report.append(f"Successful evaluations: {evaluation_results.get('successful_evaluations', 0)}")
        report.append("")

        # Key metrics
        report.append("## Key Metrics")
        for metric in ['rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'bert_score_f1']:
            mean = evaluation_results.get(f"{metric}_mean", 0)
            std = evaluation_results.get(f"{metric}_std", 0)
            report.append(f"- {metric.upper()}: {mean:.4f} ± {std:.4f}")
        report.append("")

        # Performance metrics
        report.append("## Performance Metrics")
        report.append(f"- Average generation time: {evaluation_results.get('generation_time_mean', 0):.2f}s")
        report.append(f"- Average tokens per second: {evaluation_results.get('tokens_per_second_mean', 0):.2f}")
        report.append("")

        # Detailed results
        if 'individual_results' in evaluation_results:
            report.append("## Individual Results")
            df = pd.DataFrame(evaluation_results['individual_results'])
            report.append(df.describe().to_string())

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

    def create_visualizations(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: str = "./evaluation_plots"
    ) -> List[str]:
        """
        Create visualization plots for evaluation results

        Args:
            evaluation_results: Results from evaluate_dataset
            output_dir: Directory to save plots

        Returns:
            List of created plot files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_files = []

        # Metrics distribution
        if 'individual_results' in evaluation_results:
            df = pd.DataFrame(evaluation_results['individual_results'])

            # ROUGE scores distribution
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            metrics = ['rouge_1', 'rouge_2', 'rouge_l', 'bleu']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

            for metric, (row, col) in zip(metrics, positions):
                fig.add_trace(
                    go.Histogram(x=df[metric], name=metric.upper()),
                    row=row, col=col
                )

            fig.update_layout(
                title="Evaluation Metrics Distribution",
                showlegend=False,
                height=600
            )

            plot_file = f"{output_dir}/metrics_distribution.html"
            fig.write_html(plot_file)
            plot_files.append(plot_file)

            # Performance metrics
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['generation_time'],
                y=df['tokens_per_second'],
                mode='markers',
                text=df.index,
                name='Performance'
            ))

            fig.update_layout(
                title="Generation Time vs Tokens per Second",
                xaxis_title="Generation Time (s)",
                yaxis_title="Tokens per Second",
                height=400
            )

            plot_file = f"{output_dir}/performance_scatter.html"
            fig.write_html(plot_file)
            plot_files.append(plot_file)

        return plot_files

def create_evaluation_dataset(
    prompts: List[str],
    references: List[str]
) -> Dataset:
    """
    Create a dataset for evaluation

    Args:
        prompts: List of prompts
        references: List of reference answers

    Returns:
        Dataset object
    """
    if len(prompts) != len(references):
        raise ValueError("Number of prompts must match number of references")

    data = [{"prompt": p, "reference": r} for p, r in zip(prompts, references)]
    return Dataset.from_list(data)

def load_evaluation_dataset(file_path: str) -> Dataset:
    """
    Load evaluation dataset from file

    Args:
        file_path: Path to JSON file with evaluation data

    Returns:
        Dataset object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    return Dataset.from_list(data)

# Example usage and testing
if __name__ == "__main__":
    # Example evaluation
    evaluator = QwenEvaluator("./outputs")

    # Sample evaluation data
    prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning.",
        "What are the benefits of renewable energy?"
    ]

    references = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Renewable energy sources like solar and wind power are sustainable, reduce greenhouse gas emissions, and decrease dependence on fossil fuels."
    ]

    # Run evaluation
    results = evaluator.evaluate_batch(prompts, references)

    # Print results
    for i, (prompt, reference, metrics) in enumerate(zip(prompts, references, results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Reference: {reference}")
        print(f"ROUGE-1: {metrics.rouge_1:.4f}")
        print(f"ROUGE-2: {metrics.rouge_2:.4f}")
        print(f"BLEU: {metrics.bleu:.4f}")
        print(f"Generation time: {metrics.generation_time:.2f}s")
