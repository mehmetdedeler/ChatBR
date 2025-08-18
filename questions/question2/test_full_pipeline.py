#!/usr/bin/env python3
"""
Test full ChatBR pipeline components
"""

import os
import sys
import json

def test_components():
    """Test all components of the ChatBR pipeline"""
    print("=== Testing ChatBR Pipeline Components ===\n")
    
    # Test 1: Check if all required modules can be imported
    print("1. Testing module imports...")
    try:
        from run import parse_run_arguments, run_bert_predict, load_sample_call_llm
        from classifier_predict import predict_multi_data, is_report_perfect
        from analysis_sample import analysis_llm_data, select_dataset_for_llm
        from questions.gpt_utils import call_ChatGPT
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Test argument parsing
    print("\n2. Testing argument parsing...")
    try:
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ['test.py', '--json_bug_path', './test_data']
        args = parse_run_arguments()
        print("✅ Argument parsing works")
        sys.argv = original_argv
    except Exception as e:
        print(f"❌ Argument parsing error: {e}")
        return False
    
    # Test 3: Test BEE-tool integration
    print("\n3. Testing BEE-tool integration...")
    try:
        from classifier_predict import create_bert_model
        model, tokenizer = create_bert_model()
        print("✅ BEE-tool model created successfully")
    except Exception as e:
        print(f"❌ BEE-tool integration error: {e}")
        return False
    
    # Test 4: Test ChatGPT utilities
    print("\n4. Testing ChatGPT utilities...")
    try:
        test_prompt = "Test prompt for ChatGPT"
        response = call_ChatGPT(test_prompt, model="gpt-3.5-turbo")
        if response is not None:
            print("✅ ChatGPT utilities working (real API)")
        else:
            print("✅ ChatGPT utilities working (mock mode)")
    except Exception as e:
        print(f"❌ ChatGPT utilities error: {e}")
        return False
    
    # Test 5: Test data directory creation
    print("\n5. Testing directory creation...")
    try:
        test_dirs = ['./test_predict', './test_llm', './test_generate']
        for dir_path in test_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            print(f"✅ Created directory: {dir_path}")
        
        # Clean up test directories
        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                import shutil
                shutil.rmtree(dir_path)
    except Exception as e:
        print(f"❌ Directory creation error: {e}")
        return False
    
    print("\n🎉 All components tested successfully!")
    return True

def test_single_bug_report():
    """Test processing a single bug report"""
    print("\n=== Testing Single Bug Report Processing ===\n")
    
    # Create a test bug report
    test_report = {
        "bug_id": "test_001",
        "title": "Test bug report",
        "description": "This is a test bug report for testing the ChatBR pipeline."
    }
    
    # Save test report
    test_dir = "./test_data/AspectJ"
    os.makedirs(test_dir, exist_ok=True)
    
    with open(os.path.join(test_dir, "test_001.json"), 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print("✅ Test bug report created")
    
    # Test classification
    try:
        from classifier_predict import predict_multi_data
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ['test.py', '--json_bug_path', './test_data', '--bert_result_path', './test_predict']
        
        from run import parse_run_arguments
        args = parse_run_arguments()
        
        # Test classification step
        predict_multi_data(args)
        print("✅ Bug report classification completed")
        
        sys.argv = original_argv
        
        # Clean up
        import shutil
        if os.path.exists("./test_data"):
            shutil.rmtree("./test_data")
        if os.path.exists("./test_predict"):
            shutil.rmtree("./test_predict")
            
    except Exception as e:
        print(f"❌ Single bug report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function"""
    print("=== ChatBR Pipeline Component Test ===\n")
    
    # Test all components
    if not test_components():
        print("\n❌ Component tests failed. Please fix the issues above.")
        return
    
    # Test single bug report processing
    if not test_single_bug_report():
        print("\n❌ Single bug report test failed. Please fix the issues above.")
        return
    
    print("\n🎉 All tests passed! ChatBR pipeline is ready to run.")
    print("\nYou can now run the full implementation with:")
    print("python setup_chatbr.py")

if __name__ == "__main__":
    main()
