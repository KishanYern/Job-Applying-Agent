import os
import sys

# Add the parent directory to the path so we can import from apply_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from apply_agent import _init_llm
except ImportError:
    print("[ERROR] Could not import _init_llm. Make sure you are running from the Job-Applying-Agent directory.")
    sys.exit(1)

from dotenv import load_dotenv

def test_runpod_inference():
    print("Loading environment variables from .env...")
    load_dotenv()
    
    endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    
    # Check for placeholder values
    if "<your-pod-id>" in endpoint_url or "<your_runpod_api_key>" in api_key:
        print("\n[ERROR] You still have placeholder values in your .env file!")
        print("Please replace them with your actual RunPod credentials before running this test.")
        print(f"Current RUNPOD_ENDPOINT_URL: {endpoint_url}")
        print(f"Current RUNPOD_API_KEY: {api_key}")
        return
        
    if not endpoint_url or not api_key:
        print("\n[ERROR] Missing RUNPOD_ENDPOINT_URL or RUNPOD_API_KEY in .env file.")
        return

    print(f"\nInitializing LLM with endpoint: {endpoint_url}")
    try:
        llm = _init_llm()
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        return
        
    print("\nSending a test prompt to the model...")
    print("WARNING: This may take up to 10-20 minutes if the GPU is starting cold (downloading the 32B weights).")
    
    try:
        timeout_val = getattr(llm, 'timeout', 'Default')
        print(f"Timeout is configured to: {timeout_val} seconds.")
        
        response = llm.invoke("Hello, from the Job-Applying-Agent test script! Summarize in 10 words or less what you can do.")
        
        print("\n" + "="*30)
        print("🎉 RESPONSE RECEIVED SUCCESSFULLY! 🎉")
        print("="*30)
        print(f"Content: {response.content}")
        print("="*30)
        
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        print("\nTroubleshooting Checks:")
        print("1. Is your RunPod serverless endpoint currently active/running?")
        print("2. Are there enough GPU resources available on your pod?")
        print("3. Check the RunPod server logs (on their dashboard) for backend errors.")

if __name__ == "__main__":
    test_runpod_inference()
