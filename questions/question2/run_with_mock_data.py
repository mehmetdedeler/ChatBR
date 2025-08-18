#!/usr/bin/env python3
"""
Run ChatBR with mock data using the original run.py logic
"""

import os
import json
import shutil
import sys

def create_mock_data():
    """Create simple mock bug report data"""
    print("=== Creating Mock Bug Reports ===\n")
    
    # Create origin_data directory
    origin_data_path = "./origin_data"
    if os.path.exists(origin_data_path):
        shutil.rmtree(origin_data_path)
    os.makedirs(origin_data_path)
    
    # Simple mock bug reports - just one project for testing
    projects = ["AspectJ"]  # Simplified to just one project
    
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
        
        print(f"✅ Created {project} directory with {len(sample_bug_reports)} bug reports")
    
    print(f"\n✅ Mock data created in: {origin_data_path}")
    return projects

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

def run_chatbr():
    """Run the original ChatBR implementation"""
    print("=== Running Original ChatBR Implementation ===\n")
    
    try:
        # Import and run the original run.py logic
        from run import parse_run_arguments, transfer_datatype, run_bert_predict, load_sample_call_llm
        
        # Parse arguments
        args = parse_run_arguments()
        
        # Run the original pipeline
        transfer_datatype(args)
        run_bert_predict(args)
        load_sample_call_llm(args)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running ChatBR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("=== ChatBR with Mock Data ===\n")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Step 2: Create mock data
    projects = create_mock_data()
    
    # Step 3: Run ChatBR
    print("\n" + "="*50)
    success = run_chatbr()
    
    if success:
        print("\n🎉 ChatBR implementation completed successfully!")
        print("\nResults are saved in:")
        print("- Classification results: ./predict_data/bee_tool/")
        print("- LLM dataset: ./llm_dataset/")
        print("- Generated improvements: ./generate_data/")
        
        print("\n📋 Summary:")
        print(f"- Input: {len(projects)} project(s) with mock bug reports")
        print("- Process: BEE-tool classification + ChatGPT generation")
        print("- Output: Improved bug reports with OB/EB/SR sections")
    else:
        print("\n❌ ChatBR implementation failed.")

if __name__ == "__main__":
    main()
