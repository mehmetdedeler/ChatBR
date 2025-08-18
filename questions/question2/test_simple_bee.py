#!/usr/bin/env python3
"""
Simple BEE-tool test without psutil dependency issues
"""

import json
import os
import sys
sys.path.append('../question1')

def test_bee_tool_direct():
    """Test BEE-tool directly without Stanford CoreNLP"""
    print("=== Testing BEE-tool Direct Integration ===\n")
    
    try:
        # Import BEE-tool functions
        from bee_tool import readWords, parseSentences
        
        # Test reading dictionary
        print("Testing dictionary reading...")
        readWords()
        print("✅ Dictionary loaded successfully")
        
        # Test sentence parsing
        print("\nTesting sentence parsing...")
        test_text = "The application crashes when clicking the button. This should not happen."
        sentences = parseSentences(test_text, "test_001")
        print(f"✅ Parsed {len(sentences['modifiedSentences'])} sentences")
        
        return True
        
    except Exception as e:
        print(f"❌ BEE-tool test failed: {e}")
        return False

def test_mock_data_processing():
    """Test processing mock bug reports"""
    print("\n=== Testing Mock Data Processing ===\n")
    
    # Load a mock bug report
    mock_report = {
        "bug_id": "103157",
        "title": "Bug 103157 - after returning should not bind null as return value",
        "description": "Today I found out the following. When writing an advice as... after() returning(Object o): staticinitialization(*) { System.out.println(o); } This advice is indeed executed every time a type returns from staticinitialization. However, since there is no returned object, o is bound to null."
    }
    
    print(f"Mock bug report: {mock_report['title']}")
    print(f"Description length: {len(mock_report['description'])} characters")
    
    # Test sentence extraction
    from nltk import sent_tokenize
    sentences = sent_tokenize(mock_report['title'] + ". " + mock_report['description'])
    print(f"✅ Extracted {len(sentences)} sentences from bug report")
    
    return True

def test_classifier_integration():
    """Test the classifier integration"""
    print("\n=== Testing Classifier Integration ===\n")
    
    try:
        from classifier_predict import isEmptySentence, parseReportSentences
        
        # Test sentence validation
        test_sentences = [
            "The application crashes.",
            "This is a test.",
            "a",  # Should be empty
            "   ",  # Should be empty
        ]
        
        for i, sentence in enumerate(test_sentences):
            is_empty = isEmptySentence(sentence)
            print(f"Sentence {i+1}: '{sentence}' -> Empty: {is_empty}")
        
        # Test report parsing
        test_report = "The application crashes when clicking the button. This should not happen. To reproduce: 1. Open the app 2. Click the button."
        parsed_sentences = parseReportSentences(test_report)
        print(f"\n✅ Parsed {len(parsed_sentences)} sentences from test report")
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== ChatBR BEE-tool Simple Test ===\n")
    
    # Run tests
    tests = [
        ("BEE-tool Direct", test_bee_tool_direct),
        ("Mock Data Processing", test_mock_data_processing),
        ("Classifier Integration", test_classifier_integration)
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
        print("\n🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. The BEE-tool integration is working")
        print("2. You can now run the complete ChatBR pipeline")
        print("3. Run: python run_chatbr.py")
        print("4. Or run: python run.py")
    else:
        print("\n⚠️  Some tests failed. Basic functionality may not work properly.")

if __name__ == "__main__":
    main()
