#!/usr/bin/env python3
"""
Simplified ChatBR runner with BEE-tool integration
"""

import argparse
import json
import os
from tqdm import tqdm

from classifier_predict import predict_multi_data, is_report_perfect
from questions.gpt_utils import call_ChatGPT

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
    parser.add_argument('--skip_classification', action='store_true',
                       help='Skip classification step')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip ChatGPT generation step')
    
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

def run_classification(args):
    """Run BEE-tool classification"""
    print("=== Step 1: Running BEE-tool Classification ===")
    
    # Create project directories
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    for project in project_list:
        project_dir = os.path.join(args.bert_result_path, project)
        perfect_dir = os.path.join(project_dir, 'perfect_data')
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        if not os.path.exists(perfect_dir):
            os.makedirs(perfect_dir)
    
    # Run classification
    predict_multi_data(args)
    print("✅ Classification completed!")

def run_chatgpt_generation(args):
    """Run ChatGPT generation"""
    print("=== Step 2: Running ChatGPT Generation ===")
    
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
        filelist = os.listdir(project_llm_data_path)
        p_bar = tqdm(filelist, total=len(filelist), desc=f'{project}')
        
        for idx, filename in enumerate(p_bar):
            p_bar.set_description(f"{project}: No.{idx}")
            
            # Read bug report file
            ext = os.path.splitext(filename)[1]
            if ext == '.txt':
                with open(os.path.join(project_llm_data_path, filename), 'r') as f:
                    bug_report = f.read()
                
                # Call ChatGPT
                response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                
                # Try up to 5 times to get a good response
                for i in range(5):
                    if response is not None:
                        output_file = os.path.join(project_gen_data_path, 
                                                  f"{filename.split('.')[0]}_gpt_{i}.json")
                        with open(output_file, 'w') as f:
                            json.dump(response, f)
                        
                        # Check if response is perfect
                        if is_report_perfect(response['choices'][0]['message']['content'], args):
                            break
                    
                    response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
    
    print("✅ ChatGPT generation completed!")

def main():
    """Main function to run ChatBR"""
    print("=== ChatBR with BEE-tool Integration ===\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    create_directories(args)
    
    # Step 1: Classification (if not skipped)
    if not args.skip_classification:
        run_classification(args)
    else:
        print("⏭️  Skipping classification step")
    
    # Step 2: ChatGPT Generation (if not skipped)
    if not args.skip_generation:
        run_chatgpt_generation(args)
    else:
        print("⏭️  Skipping ChatGPT generation step")
    
    print("\n🎉 ChatBR pipeline completed successfully!")
    print(f"Results saved in: {args.gen_data_path}")

if __name__ == "__main__":
    main()
