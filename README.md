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

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.45+
- Marimo 0.8+

- **Hugging Face Token** (for model downloads and uploads)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone git@github.com:jcamier/qwen-tuning-evals.git
cd QwenEvals

uv venv
source .venv/bin/activate
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync
```

3. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

### 🔑 **Getting Your Hugging Face Token**

You'll need a Hugging Face token to download models and potentially upload your fine-tuned models. Here's how to get one:

1. **Create a Hugging Face Account**:
   - Go to [huggingface.co](https://huggingface.co)
   - Click "Sign Up" and create a free account

2. **Generate an Access Token**:
   - Log into your Hugging Face account
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Choose "Write" permission (needed for uploading models)
   - Give it a name like "Qwen Fine-tuning"
   - Click "Generate a token"
   - **Copy the token immediately** (you won't see it again!)

3. **Add Token to Your Environment**:
   ```bash
   # Edit your .env file
   nano .env

   # Add your token:
   HUGGINGFACE_HUB_TOKEN=hf_your_token_here
   ```

4. **Login via CLI (Alternative)**:
   ```bash
   # Install huggingface_hub if not already installed
   pip install huggingface_hub

   # Login with your token
   huggingface-cli login
   # Enter your token when prompted
   ```

**Why you need this token:**
- Download Qwen models from Hugging Face Hub
- Upload your fine-tuned models (optional)
- Access gated models if needed
- Avoid rate limits on model downloads

### 📊 **Getting Your Weights & Biases API Key**

Weights & Biases (wandb) is used for experiment tracking, logging training metrics, and visualizing results. Here's how to set it up:

1. **Create a Weights & Biases Account**:
   - Go to [wandb.ai](https://wandb.ai)
   - Click "Sign Up" and create a free account
   - Verify your email address

2. **Get Your API Key**:
   - Log into your wandb account
   - Go to [Settings → API Keys](https://wandb.ai/settings)
   - Click "Create new key"
   - Give it a name like "Qwen Fine-tuning"
   - **Copy the API key immediately** (you won't see it again!)

3. **Add API Key to Your Environment**:
   ```bash
   # Edit your .env file
   nano .env

   # Add your API key:
   WANDB_API_KEY=your_api_key_here
   WANDB_PROJECT=qwen-finetuning-evals
   ```

4. **Login via CLI (Alternative)**:
   ```bash
   # Login with your API key
   wandb login
   # Enter your API key when prompted
   ```

**Why you need wandb:**
- Track training progress and metrics in real-time
- Visualize loss curves and evaluation metrics
- Compare different model runs
- Share results with team members
- Automatic logging of hyperparameters and results

## 🚀 Quick Start

### 1. Prepare Your Data (Optional - Sample data included)

Create sample data or use your own:

```python
from data_preparation import create_sample_training_data, create_sample_evaluation_data

# Create sample training data
create_sample_training_data("data/training_data.txt")

# Create sample evaluation data
create_sample_evaluation_data("data/evaluation_data.json")
```

### 2. Launch the Marimo Notebook

```bash
# Easy way - launches everything
python launch.py

# Or run directly
marimo edit qwen_notebook.py
```

### 3. Run Each Cell Sequentially

The notebook is organized in steps - run each cell one at a time:

1. **Imports & Setup**: Load all required libraries
2. **Configuration**: Review the model and training parameters
3. **Load Data**: Load and chunk your training data
4. **Load Model**: Load Qwen model with LoRA configuration
5. **Tokenize Dataset**: Prepare data for training
6. **Training**: Fine-tune the model (takes several minutes)
7. **Evaluation**: Test the fine-tuned model with sample prompts
8. **Summary**: Review your results

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


