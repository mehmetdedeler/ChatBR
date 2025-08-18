#!/usr/bin/env python3
"""
Simple test script for ChatBR implementation
"""

import json
import os

def test_mock_data():
    """Test that mock data exists and is properly formatted"""
    print("=== Testing Mock Data ===\n")
    
    # Check if mock data files exist
    mock_files = [
        "origin_data/AspectJ/103157.json",
        "origin_data/AspectJ/112756.json", 
        "origin_data/Birt/101372.json",
        "origin_data/Eclipse/10277.json"
    ]
    
    all_exist = True
    for file_path in mock_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
            
            # Check JSON format
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                required_fields = ['bug_id', 'title', 'description']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"❌ {file_path} missing fields: {missing_fields}")
                    all_exist = False
                else:
                    print(f"✅ {file_path} has correct JSON format")
                    
            except json.JSONDecodeError:
                print(f"❌ {file_path} has invalid JSON format")
                all_exist = False
        else:
            print(f"❌ {file_path} does not exist")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Test that required directories can be created"""
    print("\n=== Testing Directory Structure ===\n")
    
    required_dirs = [
        "predict_data/bert_new",
        "llm_dataset", 
        "generate_data"
    ]
    
    all_created = True
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created/verified directory: {dir_path}")
        except Exception as e:
            print(f"❌ Failed to create directory {dir_path}: {e}")
            all_created = False
    
    return all_created

def test_imports():
    """Test that required modules can be imported"""
    print("\n=== Testing Imports ===\n")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import nltk
        print("✅ nltk imported successfully")
    except ImportError as e:
        print(f"❌ nltk import failed: {e}")
        return False
    
    return True

def test_bee_tool_files():
    """Test that BEE-tool files exist"""
    print("\n=== Testing BEE-tool Files ===\n")
    
    bee_files = [
        "../question1/model/bee/dict.txt",
        "../question1/model/bee/model_OB.txt",
        "../question1/model/bee/model_EB.txt", 
        "../question1/model/bee/model_SR.txt",
        "../question1/model/bee/svm_classify.exe"
    ]
    
    all_exist = True
    for file_path in bee_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} does not exist")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("=== ChatBR Implementation Test ===\n")
    
    # Run all tests
    tests = [
        ("Mock Data", test_mock_data),
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("BEE-tool Files", test_bee_tool_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! ChatBR implementation is ready to run.")
        print("\nNext steps:")
        print("1. Start Stanford CoreNLP server (if needed)")
        print("2. Run: python run_chatbr.py")
        print("3. Or run: python run.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running ChatBR.")

if __name__ == "__main__":
    main()
