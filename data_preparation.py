"""
Data Preparation Utilities for Qwen Fine-tuning

This module provides utilities for:
- Loading and preprocessing training data
- Creating training datasets from various sources
- Data validation and quality checks
- Chat template formatting
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data preparation"""
    chunk_size: int = 2000
    chunk_overlap: int = 200
    min_chunk_length: int = 100
    max_chunk_length: int = 4000
    use_chat_template: bool = True
    system_message: str = "You are a helpful assistant that provides accurate and informative responses."
    validation_split: float = 0.1
    test_split: float = 0.1

class DataPreprocessor:
    """Data preprocessing utilities for Qwen fine-tuning"""

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the data preprocessor

        Args:
            config: Data preparation configuration
        """
        self.config = config or DataConfig()

    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a file

        Args:
            file_path: Path to the text file

        Returns:
            Text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def load_from_url(self, url: str) -> str:
        """
        Load text from a URL

        Args:
            url: URL to fetch text from

        Returns:
            Text content
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Try to parse as HTML first
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()

            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Error loading from URL {url}: {e}")
            raise

    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_samples: Optional[int] = None
    ) -> str:
        """
        Load text from a HuggingFace dataset

        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to use
            text_column: Column containing text data
            max_samples: Maximum number of samples to load

        Returns:
            Concatenated text content
        """
        try:
            dataset = load_dataset(dataset_name, split=split)

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            texts = dataset[text_column]
            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            raise

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            # Only add chunks that meet minimum length requirement
            if len(chunk) >= self.config.min_chunk_length:
                chunks.append(chunk)

            start += (chunk_size - overlap)

        return chunks

    def format_chat_template(
        self,
        text: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Format text using Qwen's chat template

        Args:
            text: Input text
            system_message: Optional system message

        Returns:
            Formatted chat template
        """
        if not self.config.use_chat_template:
            return text

        system_msg = system_message or self.config.system_message

        return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    def create_training_dataset(
        self,
        text_chunks: List[str],
        add_metadata: bool = True
    ) -> Dataset:
        """
        Create a training dataset from text chunks

        Args:
            text_chunks: List of text chunks
            add_metadata: Whether to add metadata to each sample

        Returns:
            Dataset object
        """
        records = []

        for i, chunk in enumerate(text_chunks):
            # Skip chunks that are too long
            if len(chunk) > self.config.max_chunk_length:
                continue

            # Format with chat template
            formatted_text = self.format_chat_template(chunk)

            record = {
                "text": formatted_text,
                "original_text": chunk,
                "chunk_id": i,
                "chunk_length": len(chunk)
            }

            if add_metadata:
                record.update({
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                    "has_questions": "?" in chunk,
                    "has_numbers": bool(re.search(r'\d', chunk)),
                })

            records.append(record)

        return Dataset.from_list(records)

    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate a dataset and return statistics

        Args:
            dataset: Dataset to validate

        Returns:
            Validation statistics
        """
        stats = {
            "total_samples": len(dataset),
            "avg_text_length": 0,
            "min_text_length": float('inf'),
            "max_text_length": 0,
            "empty_samples": 0,
            "very_short_samples": 0,
            "very_long_samples": 0,
        }

        if len(dataset) == 0:
            return stats

        text_lengths = []

        for sample in dataset:
            text_length = len(sample["text"])
            text_lengths.append(text_length)

            if text_length == 0:
                stats["empty_samples"] += 1
            elif text_length < 50:
                stats["very_short_samples"] += 1
            elif text_length > 2000:
                stats["very_long_samples"] += 1

        if text_lengths:
            stats["avg_text_length"] = np.mean(text_lengths)
            stats["min_text_length"] = min(text_lengths)
            stats["max_text_length"] = max(text_lengths)

        return stats

    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/validation/test sets

        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train, validation, test) datasets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        # Shuffle dataset
        shuffled_dataset = dataset.shuffle(seed=random_seed)
        total_size = len(shuffled_dataset)

        # Calculate split indices
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        # Split dataset
        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))

        return train_dataset, val_dataset, test_dataset

    def save_dataset(self, dataset: Dataset, output_path: str):
        """
        Save dataset to file

        Args:
            dataset: Dataset to save
            output_path: Path to save the dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.json':
            dataset.to_json(str(output_path))
        elif output_path.suffix == '.csv':
            dataset.to_csv(str(output_path))
        elif output_path.suffix == '.parquet':
            dataset.to_parquet(str(output_path))
        else:
            # Default to JSON
            dataset.to_json(str(output_path.with_suffix('.json')))

        logger.info(f"Dataset saved to {output_path}")

    def load_dataset(self, file_path: str) -> Dataset:
        """
        Load dataset from file

        Args:
            file_path: Path to the dataset file

        Returns:
            Dataset object
        """
        file_path = Path(file_path)

        if file_path.suffix == '.json':
            return Dataset.from_json(str(file_path))
        elif file_path.suffix == '.csv':
            return Dataset.from_csv(str(file_path))
        elif file_path.suffix == '.parquet':
            return Dataset.from_parquet(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def create_evaluation_dataset(
        self,
        prompts: List[str],
        references: List[str],
        categories: Optional[List[str]] = None
    ) -> Dataset:
        """
        Create an evaluation dataset

        Args:
            prompts: List of prompts
            references: List of reference answers
            categories: Optional list of categories for each prompt

        Returns:
            Dataset object
        """
        if len(prompts) != len(references):
            raise ValueError("Number of prompts must match number of references")

        if categories and len(categories) != len(prompts):
            raise ValueError("Number of categories must match number of prompts")

        records = []
        for i, (prompt, reference) in enumerate(zip(prompts, references)):
            record = {
                "prompt": prompt,
                "reference": reference,
                "id": i
            }

            if categories:
                record["category"] = categories[i]

            records.append(record)

        return Dataset.from_list(records)

def create_sample_training_data(output_path: str = "data/sample_training_data.txt"):
    """
    Create sample training data for demonstration

    Args:
        output_path: Path to save the sample data
    """
    sample_text = """
    Machine Learning and Artificial Intelligence

    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Unlike traditional programming, where explicit instructions are provided, machine learning systems learn patterns from data.

    There are three main types of machine learning:

    1. Supervised Learning: This involves training a model on labeled data, where the correct answers are provided. Examples include classification and regression tasks.

    2. Unsupervised Learning: This involves finding patterns in data without labeled examples. Common techniques include clustering and dimensionality reduction.

    3. Reinforcement Learning: This involves training an agent to make decisions by rewarding or penalizing actions in an environment.

    Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns in data. These networks are inspired by the structure and function of the human brain.

    Applications of machine learning include:
    - Image recognition and computer vision
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    - Financial modeling

    The success of machine learning depends on several factors:
    - Quality and quantity of training data
    - Appropriate algorithm selection
    - Feature engineering
    - Model validation and testing
    - Computational resources

    Ethical considerations in machine learning include bias in algorithms, privacy concerns, and the potential for job displacement. It's important to develop and deploy machine learning systems responsibly.

    Future trends in machine learning include:
    - Automated machine learning (AutoML)
    - Explainable AI
    - Federated learning
    - Edge computing
    - Quantum machine learning

    As machine learning continues to evolve, it will play an increasingly important role in solving complex problems across various industries and domains.
    """

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text.strip())

    logger.info(f"Sample training data created at {output_path}")

def create_sample_evaluation_data(output_path: str = "data/sample_evaluation_data.json"):
    """
    Create sample evaluation data for demonstration

    Args:
        output_path: Path to save the sample evaluation data
    """
    sample_data = [
        {
            "prompt": "What is machine learning?",
            "reference": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "category": "definition"
        },
        {
            "prompt": "What are the main types of machine learning?",
            "reference": "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            "category": "classification"
        },
        {
            "prompt": "Give examples of machine learning applications.",
            "reference": "Machine learning applications include image recognition, natural language processing, recommendation systems, autonomous vehicles, medical diagnosis, and financial modeling.",
            "category": "examples"
        },
        {
            "prompt": "What factors affect machine learning success?",
            "reference": "Machine learning success depends on data quality and quantity, algorithm selection, feature engineering, model validation, and computational resources.",
            "category": "factors"
        },
        {
            "prompt": "What are the ethical considerations in machine learning?",
            "reference": "Ethical considerations in machine learning include algorithmic bias, privacy concerns, and potential job displacement. Responsible development and deployment are crucial.",
            "category": "ethics"
        }
    ]

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)

    logger.info(f"Sample evaluation data created at {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    create_sample_training_data()
    create_sample_evaluation_data()

    # Example data preprocessing
    preprocessor = DataPreprocessor()

    # Load and process training data
    text = preprocessor.load_text_file("data/sample_training_data.txt")
    chunks = preprocessor.chunk_text(text)
    dataset = preprocessor.create_training_dataset(chunks)

    # Validate dataset
    validation_stats = preprocessor.validate_dataset(dataset)
    print("Dataset validation stats:", validation_stats)

    # Save dataset
    preprocessor.save_dataset(dataset, "data/processed_training_data.json")

    print(f"Created training dataset with {len(dataset)} samples")
