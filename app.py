# test_imports.py
import sys
import subprocess

required = [
    "streamlit",
    "langchain",
    "langchain-community", 
    "langchain-core",
    "openai",
    "faiss-cpu",
    "sentence-transformers"
]

print("Checking imports...")
for package in required:
    try:
        __import__(package.replace("-", "_"))
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package}")
        print(f"   Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\n✅ All packages installed!")
