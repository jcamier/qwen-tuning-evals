#!/usr/bin/env python3
"""
Launch script for the Qwen Fine-tuning and Evaluation Framework

This script provides a simple way to launch the Marimo application
and set up the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    # Map package names to their import names
    required_packages = {
        'marimo': 'marimo',
        'torch': 'torch',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'peft': 'peft',
        'accelerate': 'accelerate',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'rouge-score': 'rouge_score',
        'sacrebleu': 'sacrebleu',
        'nltk': 'nltk',
        'tqdm': 'tqdm'
    }

    missing_packages = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True

def setup_environment():
    """Set up the environment and create necessary directories"""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Create evaluation plots directory
    plots_dir = Path("evaluation_plots")
    plots_dir.mkdir(exist_ok=True)

    print("✅ Environment setup complete")

def create_sample_data():
    """Create sample data if it doesn't exist"""
    from data_preparation import create_sample_training_data, create_sample_evaluation_data

    # Create sample training data
    training_data_path = "data/sample_training_data.txt"
    if not Path(training_data_path).exists():
        print("📝 Creating sample training data...")
        create_sample_training_data(training_data_path)
        print(f"✅ Sample training data created: {training_data_path}")

    # Create sample evaluation data
    evaluation_data_path = "data/sample_evaluation_data.json"
    if not Path(evaluation_data_path).exists():
        print("📝 Creating sample evaluation data...")
        create_sample_evaluation_data(evaluation_data_path)
        print(f"✅ Sample evaluation data created: {evaluation_data_path}")

def launch_marimo():
    """Launch the Marimo application"""
    print("🚀 Launching Qwen Fine-tuning & Evaluation Framework...")

    try:
        # Run the Marimo app
        subprocess.run([
            sys.executable, "-m", "marimo", "edit", "qwen_notebook.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Marimo: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return True

    return True

def main():
    """Main launcher function"""
    print("🎯 Qwen Fine-tuning & Evaluation Framework Launcher")
    print("=" * 50)

    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        print("\n❌ Please install missing requirements and try again.")
        sys.exit(1)

    print("✅ All requirements satisfied")

    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_environment()

    # Create sample data
    print("\n📊 Creating sample data...")
    create_sample_data()

    # Launch Marimo
    print("\n🚀 Ready to launch!")
    print("The Marimo app will open in your browser.")
    print("Press Ctrl+C to stop the application.")
    print("\n" + "=" * 50)

    launch_marimo()

if __name__ == "__main__":
    main()
