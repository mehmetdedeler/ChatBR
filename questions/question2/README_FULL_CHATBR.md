# Full ChatBR Implementation with BEE-tool

This guide will help you run the complete ChatBR (ChatGPT-based Bug Report Improver) implementation using BEE-tool as the classifier instead of BERT.

## Overview

ChatBR is a two-stage bug report improvement method:
1. **Stage 1**: BEE-tool classifies sentences into OB (Observed Behavior), EB (Expected Behavior), and SR (Steps to Reproduce)
2. **Stage 2**: ChatGPT generates missing information and improves the bug report

## Prerequisites

### 1. Stanford CoreNLP
- Download Stanford CoreNLP from: https://stanfordnlp.github.io/CoreNLP/
- Start the server:
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### 2. Python Dependencies
```bash
pip install stanfordcorenlp nltk pandas tqdm openai
```

### 3. BEE-tool Files
Ensure these files exist in `../question1/model/bee/`:
- `dict.txt` - Dictionary file
- `model_OB.txt` - OB classification model
- `model_EB.txt` - EB classification model  
- `model_SR.txt` - SR classification model
- `svm_classify.exe` - SVM classifier executable (Windows)

### 4. OpenAI API Key (Optional)
Set your OpenAI API key for real ChatGPT responses:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Option 1: Automated Setup and Run
```bash
python setup_chatbr.py
```

This script will:
- Check all prerequisites
- Create mock bug report data
- Run the full ChatBR pipeline

### Option 2: Manual Step-by-Step

#### Step 1: Prepare Data
Create your bug reports in JSON format:
```json
{
  "bug_id": "1001",
  "title": "Application crashes when clicking submit button",
  "description": "When I click the submit button on the form, the application crashes with a null pointer exception."
}
```

Place them in `./origin_data/{project_name}/` directories.

#### Step 2: Run Classification
```bash
python run.py --json_bug_path ./origin_data --bert_result_path ./predict_data/bee_tool/
```

#### Step 3: Generate Improvements
```bash
python run.py --json_bug_path ./origin_data --bert_result_path ./predict_data/bee_tool/ --llm_data_path ./llm_dataset
```

## File Structure

```
questions/question2/
├── origin_data/                    # Input bug reports
│   ├── AspectJ/
│   ├── Birt/
│   └── ...
├── predict_data/bee_tool/          # BEE-tool classification results
│   ├── AspectJ/
│   ├── Birt/
│   └── ...
├── llm_dataset/                    # Formatted data for ChatGPT
│   ├── AspectJ/
│   ├── Birt/
│   └── ...
├── generate_data/                  # ChatGPT generated improvements
│   ├── AspectJ/
│   ├── Birt/
│   └── ...
├── run.py                          # Main ChatBR runner
├── classifier_predict.py           # BEE-tool integration
├── setup_chatbr.py                 # Automated setup script
└── README_FULL_CHATBR.md           # This file
```

## Output Format

### Classification Results (`predict_data/bee_tool/`)
Each bug report is classified and restructured:
```json
{
  "id": "1001",
  "title": "Application crashes when clicking submit button",
  "description": "General description text",
  "OB": "Observed behavior sentences",
  "EB": "Expected behavior sentences", 
  "SR": "Steps to reproduce sentences"
}
```

### Generated Improvements (`generate_data/`)
ChatGPT generates improved versions:
```json
{
  "choices": [{
    "message": {
      "content": "{\"id\": \"1001\", \"title\": \"Improved title\", \"description\": \"...\", \"OB\": \"...\", \"EB\": \"...\", \"SR\": \"...\"}"
    }
  }]
}
```

## Troubleshooting

### Stanford CoreNLP Issues
- **Connection refused**: Make sure Stanford CoreNLP server is running on port 9000
- **TimeExpressionExtractorImpl error**: The code now uses simplified annotation without NER

### BEE-tool Issues
- **File not found**: Check that all BEE-tool files exist in `../question1/model/bee/`
- **Permission denied**: Ensure `svm_classify.exe` has execute permissions
- **Path with spaces**: The code now sanitizes file paths to handle spaces

### ChatGPT Issues
- **API key not set**: Will use mock responses instead of real ChatGPT
- **Rate limiting**: The code includes retry logic for API failures

## Customization

### Using Your Own Data
Replace the mock data in `origin_data/` with your actual bug reports.

### Different Projects
Modify the `project_list` in `run.py` to include your projects.

### ChatGPT Model
Change the model in `load_sample_call_llm()` function:
```python
response = call_ChatGPT(bug_report, model="gpt-4")  # or "gpt-3.5-turbo"
```

## Comparison with Original ChatBR

This implementation differs from the original in one key way:
- **Original**: Uses fine-tuned BERT model for classification
- **This version**: Uses BEE-tool (SVM-based) for classification

Both versions use the same ChatGPT generation stage, making this a fair comparison for your bug report improver evaluation.

## Results Analysis

After running the pipeline, you can:
1. Compare original vs improved bug reports in `generate_data/`
2. Analyze classification quality in `predict_data/bee_tool/`
3. Use the results for your comparison study

The implementation maintains the same output format as the original ChatBR, ensuring compatibility with your evaluation framework.
