import sys
import subprocess
import importlib.util
import shutil
import os

# Define the mapping between import names and pip package names
# Format: 'import_name': 'package-name'
DEPENDENCIES = {
    'flask': 'flask',
    'flask_session': 'flask-session',
    'langchain': 'langchain',
    'langchain_community': 'langchain-community',
    'langchain_huggingface': 'langchain-huggingface',
    'langchain_ollama': 'langchain-ollama',
    'sentence_transformers': 'sentence-transformers',
    'faiss': 'faiss-cpu',
    'pdfplumber': 'pdfplumber',
    'pandas': 'pandas',
    'tabula': 'tabula-py',
    'camelot': 'camelot-py',
    'fitz': 'pymupdf',
    'easyocr': 'easyocr',
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'matplotlib': 'matplotlib',
    'networkx': 'networkx',
    'keybert': 'keybert',
    'spacy': 'spacy',
    'torch': 'torch'
}

def check_package(import_name):
    """Check if a package can be imported."""
    try:
        if import_name == 'cv2':
            import cv2
            return True
        elif import_name == 'PIL':
            import PIL
            return True
        elif import_name == 'fitz':
            import fitz
            return True
        elif import_name == 'tabula':
            import tabula
            return True
        
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except ImportError:
        return False
    except Exception:
        # Some packages might have complex init that fails if dependencies are missing
        return False

def install_packages(packages):
    """Install a list of packages using pip."""
    if not packages:
        return True

    print(f"\n📦 Installing {len(packages)} missing packages...")
    
    # Construct pip command
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Installation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed with error code {e.returncode}")
        return False

def check_system_requirements():
    """Check for system-level dependencies like Ollama."""
    print("\n🔍 Checking system requirements...")
    
    is_windows = os.name == 'nt'
    
    # Check for Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        print(f"✅ Ollama found: {ollama_path}")
        return True
    else:
        print("❌ Ollama not found in PATH.")
        if is_windows:
            print("  👉 Please install Ollama from https://ollama.com/")
        else:
            print("  👉 To install on Linux, run: curl -fsSL https://ollama.com/install.sh | sh")
        return False

def main():
    print("="*60)
    print("🚀 Catapult Chatbot Environment Initializer")
    print("="*60)
    
    missing_packages = []
    
    # 1. Check Python Dependencies
    print("\n🔍 Checking Python dependencies...")
    for import_name, package_name in DEPENDENCIES.items():
        if check_package(import_name):
            print(f"  ✅ {package_name} ({import_name}) found")
        else:
            print(f"  ❌ {package_name} missing")
            missing_packages.append(package_name)
            
    # 2. Check System Requirements
    check_system_requirements()
            
    # 3. Handle Missing Packages
    if missing_packages:
        print(f"\n⚠️  {len(missing_packages)} required packages are missing.")
        print("Missing:", ", ".join(missing_packages))
        
        while True:
            response = input("\nDo you want to install them now? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                try:
                    success = install_packages(missing_packages)
                    if not success:
                        print("Please verify the error logs above and try installing manually.")
                        sys.exit(1)
                    break
                except Exception as e:
                    print(f"Error during installation: {e}")
                    sys.exit(1)
            elif response in ['n', 'no']:
                print("\n❌ Cannot proceed without dependencies. Existing.")
                sys.exit(1)
            else:
                print("Please answer 'y' or 'n'")
    else:
        print("\n✨ All Python dependencies are satisfied!")

    # 4. Final verification and Model Check
    print("\n⏳ verifying environment...")
    
    # Check for spaCy model
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("  ✅ spaCy model 'en_core_web_sm' found")
        except OSError:
            print("  ⚠️  spaCy model 'en_core_web_sm' missing")
            response = input("  Install spaCy model 'en_core_web_sm'? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    except Exception:
        pass

    print("\n" + "="*60)
    print("✅ System Ready to Use!")
    print("="*60)
    print("\nYou can now start the application using:")
    print(f"  python app.py")

if __name__ == "__main__":
    main()
