#!/usr/bin/env python
"""
Simple test script to verify API keys and dependencies
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing API keys and dependencies...")

# Check OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in environment or .env file")
    print("   Please set it and try again")
else:
    print("✓ OPENAI_API_KEY found")
    
    # Test OpenAI API
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        # Simple model list test
        response = openai.models.list()
        print(f"✓ OpenAI API connection successful. Available models: {len(response.data)}")
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {str(e)}")

# Check Cohere API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    print("❌ COHERE_API_KEY not found in environment or .env file")
    print("   Please set it and try again")
else:
    print("✓ COHERE_API_KEY found")
    
    # Test Cohere API
    try:
        import cohere
        co = cohere.Client(COHERE_API_KEY)
        
        # Simple test
        response = co.generate(prompt="Hello, world!")
        print(f"✓ Cohere API connection successful")
    except Exception as e:
        print(f"❌ Error testing Cohere API: {str(e)}")

# Check Python dependencies
dependencies = [
    "pandas", "numpy", "pymilvus", "google-cloud-bigquery", 
    "fastapi", "uvicorn", "llama-index", "tenacity"
]

print("\nChecking Python dependencies:")
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep} installed")
    except ImportError:
        print(f"❌ {dep} not installed. Try: pip install {dep}")

print("\nTest completed")
