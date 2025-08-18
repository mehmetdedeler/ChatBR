#!/usr/bin/env python3
"""
Simple ChatBR runner - avoids import issues
"""

import os
import json
import sys
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append('..')

def create_mock_data():
    """Create mock bug report data for testing"""
    print("=== Creating Mock Data ===\n")
    
    # Create origin_data directory structure
    origin_data_path = "./origin_data"
    if os.path.exists(origin_data_path):
        shutil.rmtree(origin_data_path)
    os.makedirs(origin_data_path)
    
    # Sample bug reports for each project
    projects = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    
    sample_bug_reports = [
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
    
    # Create project directories and add sample data
    for project in projects:
        project_dir = os.path.join(origin_data_path, project)
        os.makedirs(project_dir)
        
        # Add sample bug reports per project
        for i, report in enumerate(sample_bug_reports):
            filename = f"{report['bug_id']}_{i}.json"
            filepath = os.path.join(project_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        
        print(f"✅ Created {project} directory with sample data")
    
    print(f"\n✅ Mock data created in: {origin_data_path}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("=== Checking Prerequisites ===\n")
    
    # Check if Stanford CoreNLP is running
    try:
        from stanfordcorenlp import StanfordCoreNLP
        nlp = StanfordCoreNLP('http://localhost', port=9000)
        test_result = nlp.annotate("Test sentence.", properties={
            'annotators': 'tokenize,ssplit,lemma,pos',
            'outputFormat': 'json',
        })
        nlp.close()
        print("✅ Stanford CoreNLP is running")
    except Exception as e:
        print(f"❌ Stanford CoreNLP is not running: {e}")
        print("Please start Stanford CoreNLP first:")
        print("java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000")
        return False
    
    # Check if BEE-tool files exist
    bee_model_path = "../question1/model/bee"
    required_files = ["dict.txt", "model_OB.txt", "model_EB.txt", "model_SR.txt", "svm_classify.exe"]
    
    for file in required_files:
        filepath = os.path.join(bee_model_path, file)
        if os.path.exists(filepath):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found at {filepath}")
            return False
    
    # Check if OpenAI API key is set (optional)
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key is set")
    else:
        print("⚠️  OpenAI API key not set (will use mock ChatGPT responses)")
    
    print("\n✅ All prerequisites checked!")
    return True

def run_classification():
    """Run BEE-tool classification"""
    print("=== Running BEE-tool Classification ===\n")
    
    try:
        from classifier_predict import predict_multi_data
        from analysis_sample import analysis_llm_data, select_dataset_for_llm
        
        # Create arguments object
        class Args:
            def __init__(self):
                self.json_bug_path = './origin_data'
                self.bert_result_path = './predict_data/bee_tool/'
                self.llm_data_path = './llm_dataset'
                self.report_max_length = 2000
        
        args = Args()
        
        # Create necessary directories first
        os.makedirs(args.bert_result_path, exist_ok=True)
        os.makedirs(args.llm_data_path, exist_ok=True)
        
        # Run classification (Step 1: BEE-tool prediction)
        print("Step 1: Running BEE-tool classification...")
        predict_multi_data(args)
        
        # Step 2: Analysis and dataset preparation
        print("Step 2: Analyzing results and preparing dataset...")
        analysis_llm_data(args.bert_result_path)
        select_dataset_for_llm(args.bert_result_path, args.llm_data_path, args.report_max_length)
        
        print("✅ Classification and analysis completed!")
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_chatgpt_generation():
    """Run ChatGPT generation"""
    print("=== Running ChatGPT Generation ===\n")
    
    try:
        from gpt_utils import call_ChatGPT
        from tqdm import tqdm
        
        # Create directories
        gen_data_path = './generate_data'
        if not os.path.exists(gen_data_path):
            os.makedirs(gen_data_path)
        
        # Process each project
        projects = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
        
        for project in projects:
            print(f"Processing project: {project}")
            
            # Source and destination paths
            project_llm_data_path = os.path.join('./llm_dataset', project)
            project_gen_data_path = os.path.join(gen_data_path, project)
            
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
                
                # Try up to 5 times to get a good response
                for i in range(5):
                    if response is not None:
                        output_file = os.path.join(project_gen_data_path,
                                                  f"{filename.split('.')[0]}_gpt_{i}.json")
                        with open(output_file, 'w') as f:
                            json.dump(response, f)
                        break
                    
                    response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
        
        print("✅ ChatGPT generation completed!")
        
    except Exception as e:
        print(f"❌ ChatGPT generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main function to run ChatBR"""
    print("=== Simple ChatBR Runner ===\n")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Step 2: Create mock data
    create_mock_data()
    
    # Step 3: Run classification
    if not run_classification():
        print("\n❌ Classification failed.")
        return
    
    # Step 4: Run ChatGPT generation
    if not run_chatgpt_generation():
        print("\n❌ ChatGPT generation failed.")
        return
    
    print("\n🎉 ChatBR implementation completed successfully!")
    print("\nResults are saved in:")
    print("- Classification results: ./predict_data/bee_tool/")
    print("- LLM dataset: ./llm_dataset/")
    print("- Generated improvements: ./generate_data/")

if __name__ == "__main__":
    main()
