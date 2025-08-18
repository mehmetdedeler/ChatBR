#!/usr/bin/env python3
"""
Minimal ChatBR runner - follows original run.py structure
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append('..')

from classifier_predict import predict_multi_data, is_report_perfect
from analysis_sample import analysis_llm_data, select_dataset_for_llm
from gpt_utils import call_ChatGPT

def create_minimal_mock_data():
    """Create minimal mock data for testing"""
    print("=== Creating Minimal Mock Data ===\n")
    
    # Create just one project with a few bug reports
    origin_data_path = "./origin_data"
    if os.path.exists(origin_data_path):
        import shutil
        shutil.rmtree(origin_data_path)
    os.makedirs(origin_data_path)
    
    # Create AspectJ project directory
    aspectj_dir = os.path.join(origin_data_path, "AspectJ")
    os.makedirs(aspectj_dir)
    
    # Create 3 simple bug reports
    bug_reports = [
        {
            "bug_id": "1001",
            "title": "Application crashes when clicking submit button",
            "description": "When I click the submit button on the form, the application crashes with a null pointer exception. This happens every time I try to submit the form."
        },
        {
            "bug_id": "1002", 
            "title": "Login page shows error message",
            "description": "The login page displays an error message saying 'Invalid credentials' even when I enter the correct username and password. The form should validate the credentials properly."
        },
        {
            "bug_id": "1003",
            "title": "Data not saving to database",
            "description": "When I fill out the form and click save, the data is not being saved to the database. The save operation should store the data but it appears to fail silently."
        }
    ]
    
    # Save bug reports
    for i, report in enumerate(bug_reports):
        filename = f"{report['bug_id']}_{i}.json"
        filepath = os.path.join(aspectj_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    print(f"✅ Created minimal mock data: {len(bug_reports)} bug reports in AspectJ")
    print(f"✅ Mock data created in: {origin_data_path}")

def parse_run_arguments():
    """Parse arguments like original run.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_data_path', default='./dummy_pkl/',
                        help='Original pkl file path (not used for mock data)')
    parser.add_argument('--json_bug_path', default='./origin_data/',
                        help='JSON bug report path')
    parser.add_argument('--bert_result_path', default='./predict_data/bee_tool/', 
                        help='BEE-tool prediction result path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--model_path', default='../question1/model/bee')
    parser.add_argument('--llm_data_path', default='./llm_dataset')
    parser.add_argument('--report_max_length', default=2000)

    arguments = parser.parse_args()
    return arguments

def transfer_datatype(args):
    """Transfer data type (skip for mock data)"""
    print("Skipping data transfer (using mock JSON data directly)")

def run_bert_predict(args):
    """Run BEE-tool prediction like original run.py"""
    print("=== Running BEE-tool Prediction ===\n")
    
    # Create necessary directories
    os.makedirs(args.bert_result_path, exist_ok=True)
    os.makedirs(args.llm_data_path, exist_ok=True)
    
    # Step 1: Predict bug reports using BEE-tool
    print("Step 1: Running BEE-tool classification...")
    predict_multi_data(args)
    
    # Step 2: Analyze results
    print("Step 2: Analyzing results...")
    analysis_llm_data(args.bert_result_path)
    
    # Step 3: Prepare dataset for LLM
    print("Step 3: Preparing dataset for LLM...")
    select_dataset_for_llm(args.bert_result_path, args.llm_data_path, args.report_max_length)
    
    print("✅ BEE-tool prediction completed!")

def load_sample_call_llm(args):
    """Call ChatGPT like original run.py"""
    print("=== Running ChatGPT Generation ===\n")
    
    # Create generation directory
    args.gen_data_path = './generate_data'
    if not os.path.exists(args.gen_data_path):
        os.makedirs(args.gen_data_path)

    # Use only AspectJ for minimal testing
    project_list = ["AspectJ"]
    
    for project in project_list:
        print(f"Processing project: {project}")
        
        # Get file lists
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
            
            # Call ChatGPT
            response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
            
            # Try up to 3 times to get a good response
            for i in range(3):
                if response is not None:
                    output_file = os.path.join(project_gen_data_path,
                                              f"{filename.split('.')[0]}_gpt_{i}.json")
                    with open(output_file, 'w') as f:
                        json.dump(response, f)
                    
                    # Check if response is perfect
                    try:
                        if is_report_perfect(response['choices'][0]['message']['content'], args):
                            print(f"✅ Perfect response found for {filename}")
                            break
                    except:
                        pass
                    
                    break
                
                response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
    
    print("✅ ChatGPT generation completed!")

def main():
    """Main function following original run.py structure"""
    print("=== Minimal ChatBR Implementation ===\n")
    
    # Step 1: Create minimal mock data
    create_minimal_mock_data()
    
    # Step 2: Parse arguments
    args = parse_run_arguments()
    
    # Step 3: Transfer data type (skip for mock)
    transfer_datatype(args)
    
    # Step 4: Run BEE-tool prediction
    run_bert_predict(args)
    
    # Step 5: Call ChatGPT
    load_sample_call_llm(args)
    
    print("\n🎉 Minimal ChatBR implementation completed!")
    print("\nResults are saved in:")
    print("- Classification results: ./predict_data/bee_tool/")
    print("- LLM dataset: ./llm_dataset/")
    print("- Generated improvements: ./generate_data/")

if __name__ == '__main__':
    main()
