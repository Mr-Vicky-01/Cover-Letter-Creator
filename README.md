# Gemma-2B Fine-Tuned for Cover Letter Creation
![download](https://github.com/Mr-Vicky-01/tamil_summarization/assets/143078285/408187f7-ae35-4426-9e2c-728356c2f6f8)

## Overview
Gemma-2B Fine-Tuned Python Model is a deep learning model based on the Gemma-2B architecture, fine-tuned specifically for Python programming tasks. This model is designed to understand Python code and assist developers by providing suggestions, completing code snippets, or offering corrections to improve code quality and efficiency.

## Model Details
- **Model Name**: Gemma-2B Fine-Tuned for Cover Letter Creation
- **Model Type**: Deep Learning Model
- **Base Model**: Gemma-2B
- **Language**: Python
- **Task**: Creating Cover Letter

## How to Use
1. **Install Gemma Python Package**:
   ```bash
    pip install -q -U transformers==4.38.0
    pip install torch
   ```

## Inference
1. **How to use the model in our notebook**:
```python
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Mr-Vicky-01/Gemma2B-Finetuned-CoverLetter")
model = AutoModelForCausalLM.from_pretrained("Mr-Vicky-01/Gemma2B-Finetuned-CoverLetter")

job_title = "ML Engineer"
preferred_qualification = "strong AI realted skills"
hiring_company_name = "Google"
user_name = "Vicky"
past_working_experience= "N/A"
current_working_experience = "Fresher"
skilleset= "Machine Learning, Deep Learning, AI, SQL, NLP"
qualification = "Bachelor of commerce with computer application"


prompt_template = f"<start_of_turn>user Generate Cover Letter for Role: {job_title}, \
 Preferred Qualifications: {preferred_qualification}, \
 Hiring Company: {hiring_company_name}, User Name: {user_name}, \
 Past Working Experience: {past_working_experience}, Current Working Experience: {current_working_experience}, \
 Skillsets: {skilleset}, Qualifications: {qualification} <end_of_turn>\n<start_of_turn>model"

prompt = prompt_template
encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = encodeds.to(device)


# Increase max_new_tokens if needed
generated_ids = model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
ans = ''
for i in tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('<end_of_turn>')[:2]:
    ans += i

# Extract only the model's answer
model_answer = ans.split("model")[1].strip()
print(model_answer)
```

## Model Link

{Hugging Face](https://huggingface.co/Mr-Vicky-01/Gemma2B-Finetuned-CoverLetter)
