#!/usr/bin/env python3
"""
Improved Setup script for AI PDF Analyzer
This script automates the setup and helps with OpenAI configuration
"""

import os
import sys
import subprocess
import getpass

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing Python packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("💡 Try running: pip install --upgrade pip")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Upload Configuration
MAX_UPLOAD_SIZE=16777216  # 16MB in bytes"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        return False  # Still need to configure
    return True

def configure_openai_key():
    """Help user configure OpenAI API key"""
    print("\n🔑 OpenAI API Key Configuration")
    print("=" * 40)
    
    if not os.path.exists('.env'):
        create_env_file()
    
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'OPENAI_API_KEY=your-openai-api-key-here' in content:
        print("📝 You need to configure your OpenAI API key.")
        print("\n📋 Steps to get your API key:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Sign up or log in to your OpenAI account")
        print("3. Click 'Create new secret key'")
        print("4. Copy the key (starts with 'sk-')")
        print("5. Make sure you have billing set up at: https://platform.openai.com/account/billing")
        
        choice = input("\n❓ Do you have your OpenAI API key ready? (y/n): ").lower().strip()
        
        if choice == 'y':
            api_key = getpass.getpass("🔐 Enter your OpenAI API key (hidden): ").strip()
            
            if api_key.startswith('sk-') and len(api_key) > 20:
                # Update .env file
                updated_content = content.replace('OPENAI_API_KEY=your-openai-api-key-here', f'OPENAI_API_KEY={api_key}')
                with open('.env', 'w') as f:
                    f.write(updated_content)
                print("✅ API key saved successfully!")
                return True
            else:
                print("❌ Invalid API key format. Please make sure it starts with 'sk-'")
                return False
        else:
            print("📌 No problem! You can configure it later by editing the .env file")
            return False
    else:
        print("✅ OpenAI API key appears to be configured")
        return True

def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-openai-api-key-here':
            return False
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ OpenAI API connection successful!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg:
            print("⚠️ OpenAI API key works but you have insufficient credits/quota")
            print("💡 Solution: Add payment method at https://platform.openai.com/account/billing")
        elif "401" in error_msg or "invalid" in error_msg.lower():
            print("❌ Invalid OpenAI API key")
            print("💡 Solution: Get a valid key from https://platform.openai.com/api-keys")
        else:
            print(f"⚠️ OpenAI API test failed: {error_msg}")
        return False

def create_sample_files():
    """Create sample files for testing"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Check if index.html exists
    if not os.path.exists('templates/index.html'):
        print("⚠️ templates/index.html not found")
        print("💡 Please save the index.html file in the templates/ folder")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 AI PDF Analyzer Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create .env file
    print("\n⚙️ Setting up configuration...")
    create_env_file()
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        return False
    
    # Configure OpenAI
    print("\n🤖 Configuring AI features...")
    api_configured = configure_openai_key()
    
    # Test OpenAI connection if configured
    if api_configured:
        print("\n🔍 Testing OpenAI connection...")
        connection_ok = test_openai_connection()
    else:
        connection_ok = False
    
    # Check sample files
    print("\n📄 Checking template files...")
    files_ok = create_sample_files()
    
    # Final status
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print(f"{'✅' if True else '❌'} Python dependencies installed")
    print(f"{'✅' if api_configured else '⚠️'} OpenAI API key configured")
    print(f"{'✅' if connection_ok else '⚠️'} OpenAI API connection tested")
    print(f"{'✅' if files_ok else '❌'} Template files ready")
    
    if connection_ok and files_ok:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Run: python run.py")
        print("2. Open: http://localhost:5000")
        print("3. Upload a PDF file to test")
    elif api_configured and files_ok:
        print("\n⚠️ Setup completed with warnings")
        print("\n🚀 Next steps:")
        print("1. Check your OpenAI billing at: https://platform.openai.com/account/billing")
        print("2. Add payment method if needed")
        print("3. Run: python run.py")
        print("4. Open: http://localhost:5000")
        print("\n💡 The app will work in fallback mode without full AI features")
    else:
        print("\n⚠️ Setup completed but needs attention")
        print("\n🔧 Required actions:")
        if not api_configured:
            print("- Configure OpenAI API key in .env file")
        if not files_ok:
            print("- Save index.html in templates/ folder")
        print("\n🚀 Then run: python run.py")
    
    return True

if __name__ == "__main__":
    main()