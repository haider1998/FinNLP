import os
import sys
import subprocess
from pathlib import Path


def setup_environment():
    """Setup the environment and install dependencies"""
    print("🚀 Setting up Financial Reports Analyzer...")

    # Check Python version
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        return False

    # Install dependencies
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_app():
    """Run the Streamlit application"""
    try:
        # Set environment variables for better performance
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"

        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "financial_analyzer.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")


if __name__ == "__main__":
    print("🏦 Financial Reports AI Analyzer")
    print("=" * 50)

    # Setup environment
    if setup_environment():
        print("\n🌟 Starting application...")
        run_app()
    else:
        print("❌ Setup failed. Please check the errors above.")
