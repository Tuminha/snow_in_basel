# upload_to_hf.py
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
import joblib
from datetime import datetime

# Load environment variables
load_dotenv()

def upload_model_to_huggingface():
    """
    Upload the trained snow prediction model to Hugging Face Hub
    """
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in .env file")
        return
    
    # Repository details
    repo_id = "Tuminha/snow-predictor-basel"
    
    try:
        # Initialize HF API
        api = HfApi(token=hf_token)
        
        # Create README content (use existing README.md file)
        readme_content = ""
        if os.path.exists('README.md'):
            with open('README.md', 'r', encoding='utf-8') as f:
                readme_content = f.read()
        else:
            print("⚠️ README.md not found, creating basic README")
            readme_content = """# Snow Predictor Basel

A machine learning model for predicting snow in Basel, Switzerland 7 days in advance.

## Model Files
- `snow_predictor.joblib`: Trained logistic regression model with scaler and feature names

## Usage
```python
import joblib
model_data = joblib.load('snow_predictor.joblib')
```
"""
        
        # Save README
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create a temporary folder with all files
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy model file
            model_src = "models/snow_predictor.joblib"
            model_dst = os.path.join(temp_dir, "snow_predictor.joblib")
            shutil.copy2(model_src, model_dst)
            
            # Copy README
            readme_src = "README.md"
            readme_dst = os.path.join(temp_dir, "README.md")
            shutil.copy2(readme_src, readme_dst)
            
            # Upload the entire folder
            print(f"🚀 Uploading to Hugging Face Hub: {repo_id}")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update README with proper YAML metadata"
            )
        
        print(f"✅ Model and README successfully uploaded to: https://huggingface.co/{repo_id}")
        print(f"📖 Repository: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")

if __name__ == "__main__":
    upload_model_to_huggingface()