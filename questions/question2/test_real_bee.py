#!/usr/bin/env python3
"""
Test script for real BEE-tool with Stanford CoreNLP
"""

import json
import os
import sys
import time
sys.path.append('../question1')

def test_stanford_corenlp_connection():
    """Test connection to Stanford CoreNLP"""
    print("=== Testing Stanford CoreNLP Connection ===\n")
    
    try:
        from stanfordcorenlp import StanfordCoreNLP
        
        # Try to connect to Stanford CoreNLP
        print("Attempting to connect to Stanford CoreNLP...")
        nlp = StanfordCoreNLP('http://localhost', port=9000)
        
        # Test basic functionality
        test_text = "The application crashes when clicking the button."
        result = nlp.annotate(test_text, properties={
            'annotators': 'tokenize,ssplit,lemma,pos,ner',
            'outputFormat': 'json',
        })
        
        print("✅ Successfully connected to Stanford CoreNLP!")
        print(f"✅ Test annotation result: {len(result)} characters")
        
        nlp.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Stanford CoreNLP: {e}")
        print("\nMake sure Stanford CoreNLP server is running:")
        print("1. Run: python start_stanford_corenlp.py")
        print("2. Or manually start: java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000")
        return False

def test_bee_tool_real():
    """Test real BEE-tool with Stanford CoreNLP"""
    print("\n=== Testing Real BEE-tool ===\n")
    
    try:
        from bee_tool import processText, start_nlp, close_nlp, readWords
        
        # Initialize Stanford NLP
        print("Initializing Stanford NLP...")
        nlp = start_nlp()
        
        # Read BEE-tool dictionary
        print("Reading BEE-tool dictionary...")
        readWords()
        
        # Test sentence
        test_sentence = "The application crashes when clicking the button."
        print(f"Testing with sentence: '{test_sentence}'")
        
        # Process the sentence
        result = processText("test_001", test_sentence, nlp)
        
        print("✅ BEE-tool result:")
        print(json.dumps(result, indent=2))
        
        # Extract labels
        labels = []
        for sent_data in result['bug_report'].values():
            labels.extend(sent_data['labels'])
        
        print(f"✅ Extracted labels: {labels}")
        
        # Close Stanford NLP
        close_nlp(nlp)
        
        return True
        
    except Exception as e:
        print(f"❌ BEE-tool test failed: {e}")
        return False

def test_classifier_integration_real():
    """Test the real classifier integration"""
    print("\n=== Testing Real Classifier Integration ===\n")
    
    try:
        from classifier_predict import create_bert_model, predict
        
        # Test creating BEE-tool model
        print("Creating BEE-tool model...")
        
        class MockArgs:
            def __init__(self):
                self.device = 'cpu'
                self.model_path = '../question1/model/bert-tuning-01'
                self.num_labels = 3
                self.hidden_dropout_prob = 0.3
                self.max_length = 128
        
        args = MockArgs()
        model, tokenizer = create_bert_model(args)
        
        print("✅ BEE-tool model created successfully")
        
        # Test sentence prediction
        test_sentence = "The application crashes when clicking the button."
        print(f"Testing prediction with: '{test_sentence}'")
        
        labels = predict(test_sentence, model, tokenizer, args)
        print(f"✅ Predicted labels: {labels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== ChatBR Real Implementation Test ===\n")
    
    # Run tests
    tests = [
        ("Stanford CoreNLP Connection", test_stanford_corenlp_connection),
        ("Real BEE-tool", test_bee_tool_real),
        ("Real Classifier Integration", test_classifier_integration_real)
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
        print("\n🎉 All tests passed! Real ChatBR implementation is working!")
        print("\nYou can now run the complete ChatBR pipeline:")
        print("1. Run: python run_chatbr.py")
        print("2. Or run: python run.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running ChatBR.")
        if not results[0][1]:  # Stanford CoreNLP failed
            print("\nTo fix Stanford CoreNLP issues:")
            print("1. Make sure Java is installed")
            print("2. Run: python start_stanford_corenlp.py")
            print("3. Wait for server to start (about 10-15 seconds)")

if __name__ == "__main__":
    main()
