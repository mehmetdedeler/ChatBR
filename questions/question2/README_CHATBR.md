# ChatBR Implementation with BEE-tool

This implementation uses BEE-tool instead of BERT for sentence classification in the ChatBR pipeline.

## Prerequisites

1. **Stanford CoreNLP** (already downloaded in `utils/stanford-corenlp-full-2018-01-31/`)
2. **Python dependencies**:
   ```bash
   pip install torch transformers openai pandas numpy scikit-learn tqdm nltk gensim scipy matplotlib stanfordcorenlp javalang
   ```

## File Structure

```
questions/question2/
├── classifier_predict.py      # Modified to use BEE-tool instead of BERT
├── run_chatbr.py             # Simplified runner script
├── test_bee_integration.py   # Test script for BEE-tool
├── origin_data/              # Mock bug report data
│   ├── AspectJ/
│   ├── Birt/
│   ├── Eclipse/
│   └── ...
├── predict_data/             # Classification results
├── llm_dataset/              # Formatted data for ChatGPT
└── generate_data/            # ChatGPT generated results
```

## Quick Start

### 1. Test BEE-tool Integration

First, test that BEE-tool is working correctly:

```bash
cd questions/question2
python test_bee_integration.py
```

### 2. Run Complete ChatBR Pipeline

```bash
# Run the complete pipeline
python run_chatbr.py

# Or run individual steps
python run_chatbr.py --skip_generation    # Only classification
python run_chatbr.py --skip_classification # Only ChatGPT generation
```

### 3. Run Original Implementation

If you want to use the original run.py:

```bash
python run.py
```

## Input Format

Your bug reports should be in JSON format:

```json
{
  "bug_id": "103157",
  "title": "Bug title",
  "description": "Bug description text..."
}
```

## Output

The system will:

1. **Classify sentences** using BEE-tool (OB/EB/SR labels)
2. **Reconstruct reports** with classified content
3. **Generate missing information** using ChatGPT
4. **Save improved reports** in `generate_data/` directory

## Mock Data

The implementation includes mock bug reports for testing:

- `origin_data/AspectJ/103157.json`
- `origin_data/AspectJ/112756.json`
- `origin_data/Birt/101372.json`
- `origin_data/Eclipse/10277.json`

## Customization

### Add Your Own Bug Reports

1. Create JSON files in the appropriate project directory under `origin_data/`
2. Follow the format shown above
3. Run the pipeline

### Modify Classification

The BEE-tool classification can be customized by modifying:
- `questions/question1/bee_tool.py` - Core BEE-tool logic
- `questions/question2/classifier_predict.py` - Integration layer

### Modify ChatGPT Prompts

ChatGPT prompts can be modified in:
- `questions/question4/prompt.json` - Different prompt templates

## Troubleshooting

### Stanford CoreNLP Issues

If you get Stanford CoreNLP errors:
1. Ensure the path in `bee_tool.py` points to your Stanford CoreNLP installation
2. Check that all JAR files are present
3. Verify Java is installed and accessible

### BEE-tool Issues

If BEE-tool fails:
1. Check that all model files are present in `questions/question1/model/bee/`
2. Verify the dictionary file `dict.txt` exists
3. Test with the integration test script

### ChatGPT Issues

If ChatGPT calls fail:
1. Check your OpenAI API key in `questions/gpt_utils.py`
2. Verify internet connectivity
3. Check API rate limits

## Performance Notes

- **BEE-tool**: Slower than BERT but no training required
- **ChatGPT**: API calls may take time depending on response length
- **Memory**: Stanford CoreNLP can be memory-intensive

## Comparison with Your Tool

To compare with your own bug report improver:

1. **Run ChatBR** on your unimproved bug reports
2. **Run your tool** on the same bug reports
3. **Compare outputs** using the same evaluation metrics
4. **Analyze differences** in OB/EB/SR generation quality

The BEE-tool approach provides a fair baseline comparison since it's the original baseline used in the ChatBR paper.
