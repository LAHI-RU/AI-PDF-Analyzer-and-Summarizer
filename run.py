#!/usr/bin/env python3
"""
Launch script for AI PDF Analyzer
"""

import os
import sys
from dotenv import load_dotenv

def check_requirements():
    """Check if all requirements are met"""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please run: python setup.py")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        print("❌ OpenAI API key not configured!")
        print("Please update your .env file with a valid OpenAI API key")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    # Check if required directories exist
    required_dirs = ['uploads', 'templates']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Directory missing: {directory}")
            print("Please run: python setup.py")
            return False
    
    return True

def main():
    """Main function to launch the application"""
    print("🤖 AI PDF Analyzer")
    print("=" * 30)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All requirements met!")
    print("🚀 Starting application...")
    print("📝 Make sure you have a valid OpenAI API key")
    print("🌐 Access the app at: http://localhost:5000")
    print("=" * 30)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure app.py exists in the current directory")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()