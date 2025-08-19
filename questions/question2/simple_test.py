#!/usr/bin/env python3
"""
Simple ChatBR test with 3 inline bug reports - no project files needed
"""

import os
import json
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append('..')

def create_temp_bug_reports():
    """Create 3 sample bug reports in a temporary directory"""
    print("=== Creating Sample Bug Reports ===\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    origin_data_path = os.path.join(temp_dir, "origin_data")
    os.makedirs(origin_data_path)
    
    # Create a simple project directory
    project_dir = os.path.join(origin_data_path, "TestProject")
    os.makedirs(project_dir)
    
    # 3 sample bug reports
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
    
    # Save bug reports
    for i, report in enumerate(sample_bug_reports):
        filename = f"{report['bug_id']}_{i}.json"
        filepath = os.path.join(project_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    print(f"✅ Created {len(sample_bug_reports)} bug reports in temporary directory")
    for report in sample_bug_reports:
        print(f"  - {report['bug_id']}: {report['title']}")
    
    return temp_dir, origin_data_path

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

def run_chatbr_with_temp_data(temp_dir, origin_data_path):
    """Run ChatBR with temporary data using the same logic as run.py"""
    print("=== Running ChatBR with Sample Data ===\n")
    
    try:
        # Import the same functions from run.py
        from run import parse_run_arguments, transfer_datatype, run_bert_predict, load_sample_call_llm
        
        # Create arguments with temporary paths
        class TempArgs:
            def __init__(self, temp_dir, origin_data_path):
                self.pkl_data_path = os.path.join(temp_dir, "mock_data/pickles/")
                self.json_bug_path = origin_data_path
                self.bert_result_path = os.path.join(temp_dir, "predict_data/bee_tool/")
                self.device = 'gpu'
                self.num_labels = 3
                self.max_length = 128
                self.hidden_dropout_prob = 0.3
                self.model_path = '../question1/model/bee'
                self.llm_data_path = os.path.join(temp_dir, "llm_dataset")
                self.report_max_length = 2000
        
        args = TempArgs(temp_dir, origin_data_path)
        
        print(f"Input data: {args.json_bug_path}")
        print(f"Output results: {args.bert_result_path}")
        print(f"LLM dataset: {args.llm_data_path}")
        print()
        
        # Run the same pipeline as run.py
        transfer_datatype(args)
        run_bert_predict(args)
        load_sample_call_llm(args)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running ChatBR: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_results(temp_dir):
    """Show the results of the ChatBR processing"""
    print("\n=== Results ===\n")
    
    # Show classification results
    predict_path = os.path.join(temp_dir, "predict_data/bee_tool/TestProject")
    if os.path.exists(predict_path):
        print("📊 Classification Results:")
        json_files = [f for f in os.listdir(predict_path) if f.endswith('.json')]
        for json_file in json_files:
            with open(os.path.join(predict_path, json_file), 'r') as f:
                report = json.load(f)
            print(f"  - {json_file}:")
            print(f"    Title: {report.get('title', 'N/A')}")
            print(f"    OB: {len(report.get('OB', ''))} chars")
            print(f"    EB: {len(report.get('EB', ''))} chars")
            print(f"    SR: {len(report.get('SR', ''))} chars")
            print()
    
    # Show generated improvements
    gen_path = os.path.join(temp_dir, "generate_data/TestProject")
    if os.path.exists(gen_path):
        print("🤖 Generated Improvements:")
        json_files = [f for f in os.listdir(gen_path) if f.endswith('.json')]
        for json_file in json_files:
            print(f"  - {json_file}")
    
    print(f"\n📁 All results saved in: {temp_dir}")

def main():
    """Main function"""
    print("=== Simple ChatBR Test with 3 Bug Reports ===\n")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Step 2: Create temporary bug reports
    temp_dir, origin_data_path = create_temp_bug_reports()
    
    try:
        # Step 3: Run ChatBR with the same logic as run.py
        print("\n" + "="*50)
        success = run_chatbr_with_temp_data(temp_dir, origin_data_path)
        
        if success:
            print("\n🎉 ChatBR test completed successfully!")
            show_results(temp_dir)
            
            print("\n📋 Summary:")
            print("- Input: 3 sample bug reports")
            print("- Process: BEE-tool classification + ChatGPT generation")
            print("- Output: Improved bug reports with OB/EB/SR sections")
            print(f"- Location: {temp_dir}")
        else:
            print("\n❌ ChatBR test failed.")
    
    finally:
        # Clean up temporary directory (comment out to keep results)
        # shutil.rmtree(temp_dir)
        print(f"\n💡 Temporary files kept in: {temp_dir}")
        print("   (Uncomment cleanup line in code to remove automatically)")

if __name__ == "__main__":
    main()
