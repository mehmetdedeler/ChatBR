#!/usr/bin/env python3
"""
Setup and run full ChatBR implementation with BEE-tool
"""

import os
import json
import shutil
from pathlib import Path

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
        },
        {
            "bug_id": "1004",
            "title": "Search functionality not working",
            "description": "The search box on the main page does not return any results when I search for existing items. The search should find and display matching results from the database."
        },
        {
            "bug_id": "1005",
            "title": "Page layout broken on mobile",
            "description": "The page layout is completely broken when viewed on mobile devices. The content should be responsive and display properly on all screen sizes."
        }
    ]
    
    # Create project directories and add sample data
    for project in projects:
        project_dir = os.path.join(origin_data_path, project)
        os.makedirs(project_dir)
        
        # Add 2-3 sample bug reports per project
        for i, report in enumerate(sample_bug_reports[:3]):
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
    import os
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key is set")
    else:
        print("⚠️  OpenAI API key not set (will use mock ChatGPT responses)")
    
    print("\n✅ All prerequisites checked!")
    return True

def run_chatbr():
    """Run the full ChatBR implementation"""
    print("=== Running Full ChatBR Implementation ===\n")
    
    # Set up arguments for run.py
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [
        'run.py',
        '--json_bug_path', './origin_data',
        '--bert_result_path', './predict_data/bee_tool/',
        '--llm_data_path', './llm_dataset',
        '--report_max_length', '2000'
    ]
    
    try:
        # Import the run module and execute its main logic
        from run import parse_run_arguments, transfer_datatype, run_bert_predict, load_sample_call_llm
        
        # Parse arguments
        args = parse_run_arguments()
        
        # Step 1: Transfer data type (skip if no pkl files)
        print("Step 1: Data preparation...")
        # transfer_datatype(args)  # Skip this step since we're using JSON directly
        
        # Step 2: Run BEE-tool classification
        print("Step 2: Running BEE-tool classification...")
        run_bert_predict(args)
        
        # Step 3: Run ChatGPT generation
        print("Step 3: Running ChatGPT generation...")
        load_sample_call_llm(args)
        
    except Exception as e:
        print(f"❌ Error running ChatBR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original argv
        sys.argv = original_argv

def main():
    """Main setup and run function"""
    print("=== ChatBR Setup and Run ===\n")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Step 2: Create mock data
    create_mock_data()
    
    # Step 3: Run ChatBR
    print("\n" + "="*50)
    run_chatbr()
    
    print("\n🎉 ChatBR implementation completed!")
    print("\nResults are saved in:")
    print("- Classification results: ./predict_data/bee_tool/")
    print("- LLM dataset: ./llm_dataset/")
    print("- Generated improvements: ./generate_data/")

if __name__ == "__main__":
    main()
