import os
import sys

# Append the project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
try:
    model = os.getenv("LLM_MODEL", "tinyllama")
    print(f"Testing Ollama with localhost (model={model})...")
    llm = Ollama(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "1024")),
    )
    response = llm.invoke("Hello")
    print(f"Success! Response: {response}")
except Exception as e:
    print(f"Error: {e}")
