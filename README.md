# Qwen Fine-Tuning & Evaluation Framework

A comprehensive framework for fine-tuning and evaluating Qwen models using Marimo.

**Inspired by Eric Livesay's talk on Fine Tuning Qwen at UTA Python meetup (October 16th, 2025)**

For more details on the original notebook: https://github.com/elivesay/elivesay.github.io/blob/main/qwen_finetuning_updated.ipynb

## 🚀 Features

- **Interactive Marimo Interface**: Modern, reactive UI for fine-tuning and evaluation
- **Comprehensive Evaluation**: Multiple metrics including ROUGE, BLEU, BERTScore, and custom metrics
- **Advanced Training**: LoRA fine-tuning with quantization support
- **Data Processing**: Flexible data loading from files, URLs, and HuggingFace datasets
- **Visualization**: Training curves, evaluation metrics, and performance analysis
- **Model Comparison**: Side-by-side comparison of different models
- **Automated Reporting**: Generate detailed evaluation reports

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.45+
- Marimo 0.8+

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone git@github.com:jcamier/qwen-tuning-evals.git
cd QwenEvals
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Set up environment variables** (optional):
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

## 🚀 Quick Start

### 1. Prepare Your Data

Create sample data or use your own:

```python
from data_preparation import create_sample_training_data, create_sample_evaluation_data

# Create sample training data
create_sample_training_data("data/training_data.txt")

# Create sample evaluation data
create_sample_evaluation_data("data/evaluation_data.json")
```

### 2. Launch the Marimo App

```bash
marimo run qwen_finetune_marimo.py
```

### 3. Configure and Run

1. **Model Configuration**: Set your base model (default: Qwen/Qwen3-0.6B)
2. **Training Parameters**: Configure epochs, learning rate, batch size
3. **LoRA Settings**: Adjust LoRA rank, alpha, and dropout
4. **Data Source**: Load your training data
5. **Evaluation**: Set up evaluation prompts and metrics
6. **Start Training**: Click "Start Training" to begin
7. **Run Evaluations**: Click "Run Evaluations" to assess performance

## 📊 Evaluation Framework

The framework provides comprehensive evaluation capabilities:

### Metrics Supported

- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU Score**: Bilingual Evaluation Understudy
- **BERTScore**: Contextual embedding-based similarity
- **Exact Match**: Perfect string matching
- **Generation Speed**: Tokens per second, generation time

### Evaluation Types

1. **Single Sample Evaluation**: Evaluate individual prompt-response pairs
2. **Batch Evaluation**: Process multiple samples efficiently
3. **Dataset Evaluation**: Comprehensive evaluation on full datasets
4. **Model Comparison**: Compare different models side-by-side

### Example Usage

```python
from evaluation_framework import QwenEvaluator, EvaluationConfig

# Initialize evaluator
evaluator = QwenEvaluator("./outputs")

# Configure evaluation
config = EvaluationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

# Evaluate single sample
metrics = evaluator.evaluate_single(
    prompt="What is machine learning?",
    reference="Machine learning is a subset of AI...",
    config=config
)

print(f"ROUGE-1: {metrics.rouge_1:.4f}")
print(f"BLEU: {metrics.bleu:.4f}")
```

## 📈 Data Preparation

The framework supports multiple data sources:

### Supported Formats

- **Text Files**: Plain text files
- **URLs**: Web pages and online content
- **HuggingFace Datasets**: Direct integration with HF datasets
- **JSON/CSV**: Structured data formats

### Data Processing

```python
from data_preparation import DataPreprocessor, DataConfig

# Configure data processing
config = DataConfig(
    chunk_size=2000,
    chunk_overlap=200,
    use_chat_template=True
)

preprocessor = DataPreprocessor(config)

# Load and process data
text = preprocessor.load_text_file("data/training_data.txt")
chunks = preprocessor.chunk_text(text)
dataset = preprocessor.create_training_dataset(chunks)

# Validate and save
stats = preprocessor.validate_dataset(dataset)
preprocessor.save_dataset(dataset, "data/processed_data.json")
```

## 🎯 Advanced Features

### Model Comparison

Compare different models or configurations:

```python
# Compare two models
comparison = evaluator.compare_models(
    other_model_path="./other_model",
    prompts=test_prompts,
    references=test_references
)

print(f"ROUGE-1 improvement: {comparison['rouge_1_improvement']:.4f}")
```

### Custom Evaluation Metrics

Add your own evaluation metrics:

```python
def custom_metric(prediction: str, reference: str) -> float:
    # Your custom metric implementation
    return similarity_score

# Use in evaluation
metrics.custom_score = custom_metric(prediction, reference)
```

### Automated Reporting

Generate comprehensive evaluation reports:

```python
# Generate report
report = evaluator.generate_report(evaluation_results, "evaluation_report.md")

# Create visualizations
plot_files = evaluator.create_visualizations(
    evaluation_results,
    "evaluation_plots/"
)
```

## 🔧 Configuration

### Training Configuration

```python
@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 2
    max_length: int = 1024
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_4bit: bool = False
    use_8bit: bool = False
```

### Evaluation Configuration

```python
@dataclass
class EvaluationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True
```

## 📚 Examples

### Example 1: Basic Fine-tuning

```python
# 1. Prepare data
preprocessor = DataPreprocessor()
text = preprocessor.load_text_file("my_data.txt")
chunks = preprocessor.chunk_text(text)
dataset = preprocessor.create_training_dataset(chunks)

# 2. Configure training
config = TrainingConfig(
    model_name="Qwen/Qwen3-0.6B",
    epochs=3,
    learning_rate=2e-5
)

# 3. Train model (using Marimo interface)
# The Marimo app handles the training process
```

### Example 2: Comprehensive Evaluation

```python
# 1. Load evaluation data
eval_data = load_evaluation_dataset("evaluation_data.json")

# 2. Initialize evaluator
evaluator = QwenEvaluator("./fine_tuned_model")

# 3. Run evaluation
results = evaluator.evaluate_dataset(eval_data)

# 4. Generate report
report = evaluator.generate_report(results, "evaluation_report.md")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading Errors**: Ensure you have the correct model path and sufficient disk space
3. **Evaluation Timeout**: Reduce max_new_tokens or use smaller evaluation sets

### Performance Tips

1. **Use Quantization**: Enable 4-bit or 8-bit quantization for memory efficiency
2. **Batch Processing**: Process evaluations in batches for better performance
3. **Caching**: Cache tokenized data to avoid reprocessing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Eric Livesay** for the inspiring talk on Fine Tuning Qwen at UTA Python meetup
- **Qwen Team** for the excellent Qwen models
- **Marimo Team** for the innovative notebook framework
- **Hugging Face** for the transformers library

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Join our community discussions

---


