#!/usr/bin/env python3
"""
Minimal ChatBR test - process simple bug reports and see improvements
"""

import os
import json
import sys
sys.path.append('..')

def create_simple_mock_data():
    """Create simple mock bug reports"""
    print("=== Creating Simple Mock Data ===\n")
    
    # Create a simple directory structure
    origin_data_path = "./simple_test_data"
    if os.path.exists(origin_data_path):
        import shutil
        shutil.rmtree(origin_data_path)
    os.makedirs(origin_data_path)
    
    # Create just one project directory
    project_dir = os.path.join(origin_data_path, "TestProject")
    os.makedirs(project_dir)
    
    # Simple bug reports
    simple_bug_reports = [
        {
            "bug_id": "001",
            "title": "Login button not working",
            "description": "When I click the login button, nothing happens. The button should take me to the dashboard."
        },
        {
            "bug_id": "002", 
            "title": "Search results are empty",
            "description": "I search for existing items but get no results. The search should show matching items from the database."
        },
        {
            "bug_id": "003",
            "title": "Form validation error",
            "description": "The form shows an error message even when I enter valid data. It should only show errors for invalid input."
        }
    ]
    
    # Save bug reports
    for report in simple_bug_reports:
        filename = f"{report['bug_id']}.json"
        filepath = os.path.join(project_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    print(f"✅ Created {len(simple_bug_reports)} simple bug reports in {origin_data_path}")
    return origin_data_path

def run_minimal_classification(data_path):
    """Run classification on simple data"""
    print("=== Running Minimal Classification ===\n")
    
    try:
        from classifier_predict import predict_multi_data
        
        # Create args object
        class Args:
            def __init__(self):
                self.json_bug_path = data_path
                self.bert_result_path = './simple_predict_data/'
                self.llm_data_path = './simple_llm_dataset/'
                self.report_max_length = 2000
        
        args = Args()
        
        # Create directories
        os.makedirs(args.bert_result_path, exist_ok=True)
        os.makedirs(args.llm_data_path, exist_ok=True)
        
        # Run classification
        predict_multi_data(args)
        print("✅ Classification completed!")
        
        return args
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_minimal_analysis(args):
    """Run analysis on simple data"""
    print("=== Running Minimal Analysis ===\n")
    
    try:
        from analysis_sample import analysis_llm_data, select_dataset_for_llm
        
        # Run analysis
        analysis_llm_data(args.bert_result_path)
        select_dataset_for_llm(args.bert_result_path, args.llm_data_path, args.report_max_length)
        
        print("✅ Analysis completed!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def run_minimal_chatgpt(args):
    """Run ChatGPT generation on simple data"""
    print("=== Running Minimal ChatGPT Generation ===\n")
    
    try:
        from gpt_utils import call_ChatGPT
        from classifier_predict import is_report_perfect
        
        # Create output directory
        gen_data_path = './simple_generate_data/'
        os.makedirs(gen_data_path, exist_ok=True)
        
        # Process the simple dataset
        project_llm_data_path = os.path.join(args.llm_data_path, "TestProject")
        project_gen_data_path = os.path.join(gen_data_path, "TestProject")
        os.makedirs(project_gen_data_path, exist_ok=True)
        
        if not os.path.exists(project_llm_data_path):
            print("Warning: No LLM data found")
            return
        
        # Process files
        filelist = [f for f in os.listdir(project_llm_data_path) if f.endswith('.txt')]
        print(f"Found {len(filelist)} files to process")
        
        for filename in filelist:
            print(f"Processing: {filename}")
            
            # Read bug report file
            with open(os.path.join(project_llm_data_path, filename), 'r') as f:
                bug_report = f.read()
            
            # Call ChatGPT
            response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
            
            if response is not None:
                output_file = os.path.join(project_gen_data_path, f"{filename.split('.')[0]}_improved.json")
                with open(output_file, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"✅ Generated improvement: {output_file}")
            else:
                print(f"⚠️  No response for {filename}")
        
        print("✅ ChatGPT generation completed!")
        
    except Exception as e:
        print(f"❌ ChatGPT generation failed: {e}")
        import traceback
        traceback.print_exc()

def show_results():
    """Show the results"""
    print("\n=== Results Summary ===\n")
    
    # Show classification results
    predict_path = "./simple_predict_data/TestProject"
    if os.path.exists(predict_path):
        files = [f for f in os.listdir(predict_path) if f.endswith('.json')]
        print(f"✅ Classification results: {len(files)} files in {predict_path}")
        
        # Show one example
        if files:
            with open(os.path.join(predict_path, files[0]), 'r') as f:
                example = json.load(f)
            print(f"Example classified report:")
            print(f"  Title: {example.get('title', 'N/A')}")
            print(f"  OB: {example.get('OB', 'N/A')[:50]}...")
            print(f"  EB: {example.get('EB', 'N/A')[:50]}...")
            print(f"  SR: {example.get('SR', 'N/A')[:50]}...")
    
    # Show generation results
    gen_path = "./simple_generate_data/TestProject"
    if os.path.exists(gen_path):
        files = [f for f in os.listdir(gen_path) if f.endswith('.json')]
        print(f"\n✅ Generated improvements: {len(files)} files in {gen_path}")
        
        # Show one example
        if files:
            with open(os.path.join(gen_path, files[0]), 'r') as f:
                example = json.load(f)
            print(f"Example improved report:")
            if 'choices' in example and len(example['choices']) > 0:
                content = example['choices'][0]['message']['content']
                print(f"  Content: {content[:100]}...")

def main():
    """Main function"""
    print("=== Minimal ChatBR Test ===\n")
    
    # Step 1: Create simple mock data
    data_path = create_simple_mock_data()
    
    # Step 2: Run classification
    args = run_minimal_classification(data_path)
    if args is None:
        return
    
    # Step 3: Run analysis
    run_minimal_analysis(args)
    
    # Step 4: Run ChatGPT generation
    run_minimal_chatgpt(args)
    
    # Step 5: Show results
    show_results()
    
    print("\n🎉 Minimal ChatBR test completed!")
    print("\nYou can now see:")
    print("- Original bug reports: ./simple_test_data/")
    print("- Classified reports: ./simple_predict_data/")
    print("- Improved reports: ./simple_generate_data/")

if __name__ == "__main__":
    main()
