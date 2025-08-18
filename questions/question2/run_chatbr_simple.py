#!/usr/bin/env python3
"""
Simplified ChatBR runner that works without Stanford CoreNLP
"""

import argparse
import json
import os
from tqdm import tqdm
from nltk import sent_tokenize

def parse_arguments():
    """Parse arguments for ChatBR"""
    parser = argparse.ArgumentParser(description='Run ChatBR with BEE-tool')
    parser.add_argument('--json_bug_path', default='./origin_data', 
                       help='Path to bug report JSON files')
    parser.add_argument('--bert_result_path', default='./predict_data/bert_new/', 
                       help='BERT/BEE prediction result path')
    parser.add_argument('--llm_data_path', default='./llm_dataset',
                       help='LLM dataset path')
    parser.add_argument('--gen_data_path', default='./generate_data',
                       help='Generated data path')
    parser.add_argument('--report_max_length', default=2000,
                       help='Maximum report length')
    
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories"""
    dirs = [
        args.bert_result_path,
        args.llm_data_path,
        args.gen_data_path
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def mock_classify_sentences(sentences):
    """Mock sentence classification (simulates BEE-tool output)"""
    # This is a simplified mock classification
    # In a real scenario, this would use BEE-tool or BERT
    labels = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_labels = []
        
        # Simple rule-based classification for demonstration
        if any(word in sentence_lower for word in ['crash', 'error', 'fail', 'bug', 'problem', 'issue']):
            sentence_labels.append('OB')
        if any(word in sentence_lower for word in ['should', 'expect', 'hope', 'need', 'want']):
            sentence_labels.append('EB')
        if any(word in sentence_lower for word in ['reproduce', 'step', 'click', 'open', 'run', 'execute']):
            sentence_labels.append('SR')
        
        labels.append(sentence_labels)
    
    return labels

def classify_bug_reports(args):
    """Classify bug reports using mock classification"""
    print("=== Step 1: Running Mock Classification ===")
    
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    
    for project in project_list:
        project_dir = os.path.join(args.bert_result_path, project)
        perfect_dir = os.path.join(project_dir, 'perfect_data')
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        if not os.path.exists(perfect_dir):
            os.makedirs(perfect_dir)
        
        # Check if project has data
        project_data_dir = os.path.join(args.json_bug_path, project)
        if not os.path.exists(project_data_dir):
            print(f"Warning: No data found for {project}")
            continue
        
        print(f"Processing project: {project}")
        
        # Process files
        filelist = [f for f in os.listdir(project_data_dir) if f.endswith('.json')]
        p_bar = tqdm(filelist, total=len(filelist), desc=f'{project}')
        
        for idx, filename in enumerate(p_bar):
            p_bar.set_description(f"{project}: No.{idx}")
            
            # Load bug report
            with open(os.path.join(project_data_dir, filename), 'r') as f:
                report = json.load(f)
            
            # Extract sentences
            text = report['title'] + ". " + report['description']
            sentences = sent_tokenize(text)
            
            # Classify sentences
            sentence_labels = mock_classify_sentences(sentences)
            
            # Create new report
            new_report = {
                "id": report["bug_id"],
                "title": report['title'],
                "description": "",
                "OB": "",
                "EB": "",
                "SR": ""
            }
            
            # Reconstruct report
            all_labels = []
            for i, sentence in enumerate(sentences):
                labels = sentence_labels[i]
                all_labels.extend(labels)
                
                if 'OB' in labels:
                    new_report["OB"] += sentence + " "
                elif 'EB' in labels:
                    new_report["EB"] += sentence + " "
                elif 'SR' in labels:
                    new_report["SR"] += sentence + " "
                else:
                    new_report["description"] += sentence + " "
            
            # Save report
            if len(set(all_labels)) == 3:  # Has OB, EB, SR
                output_file = os.path.join(perfect_dir, f"{report['bug_id']}.json")
            else:
                output_file = os.path.join(project_dir, f"{report['bug_id']}.json")
            
            with open(output_file, 'w') as f:
                json.dump(new_report, f)
    
    print("✅ Classification completed!")

def create_llm_dataset(args):
    """Create dataset for ChatGPT generation"""
    print("=== Step 1.5: Creating LLM Dataset ===")
    
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    
    for project in project_list:
        project_bert_dir = os.path.join(args.bert_result_path, project)
        project_llm_dir = os.path.join(args.llm_data_path, project)
        
        if not os.path.exists(project_llm_dir):
            os.makedirs(project_llm_dir)
        
        if not os.path.exists(project_bert_dir):
            continue
        
        # Get all JSON files
        json_files = [f for f in os.listdir(project_bert_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            with open(os.path.join(project_bert_dir, json_file), 'r') as f:
                report = json.load(f)
            
            # Create prompt for ChatGPT
            prompt = f"""Your role is a senior software engineer, you are very good at analyzing and writing bug reports.
In your bug reports, it is essential to provide clear and informative statements for the following categories:
- **Observed Behavior (OB):** This section should describe the relevant software behavior, actions, output, or results. Avoid vague statements like "the system does not work."
- **Expected Behavior (EB):** This part should articulate what the software should or is expected to do, using phrases like "should...", "expect...", or "hope...". Avoid suggestions or recommendations for bug resolution in this section.
- **Steps to Reproduce (SR):** Include user actions or operations that can potentially lead to reproducing the issue. Use phrases like "to reproduce," "steps to reproduce," or "follow these steps."

It is possible that the bug report may lack sufficient details in the OB, EB, and SR sections. In such cases, your task is to infer the appropriate details based on the context and supplement the bug report to ensure it contains clear and complete OB/EB/SR statements. Also, improve the wording of these statements for clarity where possible.

To facilitate this process, please provide your responses in JSON format as follows:
{{"id": "", "title": "", "description": "", "OB": "", "EB": "", "SR": ""}}

<BUG REPORT>
{json.dumps(report)}
</BUG REPORT>"""
            
            # Save prompt
            output_file = os.path.join(project_llm_dir, f"{report['id']}.txt")
            with open(output_file, 'w') as f:
                f.write(prompt)
    
    print("✅ LLM dataset created!")

def run_chatgpt_generation(args):
    """Run ChatGPT generation (mock version)"""
    print("=== Step 2: Running ChatGPT Generation (Mock) ===")
    
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    
    for project in project_list:
        print(f"Processing project: {project}")
        
        # Source and destination paths
        project_llm_data_path = os.path.join(args.llm_data_path, project)
        project_gen_data_path = os.path.join(args.gen_data_path, project)
        
        if not os.path.exists(project_gen_data_path):
            os.makedirs(project_gen_data_path)
        
        if not os.path.exists(project_llm_data_path):
            print(f"Warning: No data found for {project}")
            continue
        
        # Process files
        filelist = [f for f in os.listdir(project_llm_data_path) if f.endswith('.txt')]
        p_bar = tqdm(filelist, total=len(filelist), desc=f'{project}')
        
        for idx, filename in enumerate(p_bar):
            p_bar.set_description(f"{project}: No.{idx}")
            
            # Read bug report file
            with open(os.path.join(project_llm_data_path, filename), 'r') as f:
                bug_report = f.read()
            
            # Mock ChatGPT response (in real scenario, this would call the API)
            mock_response = {
                "choices": [{
                    "message": {
                        "content": '{"id": "mock_id", "title": "Mock Title", "description": "Mock description", "OB": "Mock observed behavior", "EB": "Mock expected behavior", "SR": "Mock steps to reproduce"}'
                    }
                }],
                "finish_reason": "stop"
            }
            
            # Save response
            output_file = os.path.join(project_gen_data_path, 
                                      f"{filename.split('.')[0]}_gpt_0.json")
            with open(output_file, 'w') as f:
                json.dump(mock_response, f)
    
    print("✅ ChatGPT generation completed (mock)!")

def main():
    """Main function to run ChatBR"""
    print("=== ChatBR with Mock BEE-tool Integration ===\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    create_directories(args)
    
    # Step 1: Classification
    classify_bug_reports(args)
    
    # Step 1.5: Create LLM dataset
    create_llm_dataset(args)
    
    # Step 2: ChatGPT Generation
    run_chatgpt_generation(args)
    
    print("\n🎉 ChatBR pipeline completed successfully!")
    print(f"Results saved in: {args.gen_data_path}")
    print("\nNote: This is a mock implementation. For real ChatGPT calls:")
    print("1. Update your OpenAI API key in questions/gpt_utils.py")
    print("2. Replace mock_chatgpt_generation with real API calls")

if __name__ == "__main__":
    main()
