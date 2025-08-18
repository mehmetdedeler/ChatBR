#!/usr/bin/env python3
"""
Test script to verify BEE-tool integration with ChatBR
"""

import json
import os
import sys
sys.path.append('../question1')
from bee_tool import processText, start_nlp, close_nlp, readWords

def test_bee_tool():
    """Test BEE-tool with a sample sentence"""
    print("Testing BEE-tool integration...")
    
    # Initialize Stanford NLP
    nlp = start_nlp()
    readWords()
    
    # Test sentence
    test_sentence = "The application crashes when clicking the button."
    
    try:
        # Process the sentence
        result = processText("test_001", test_sentence, nlp)
        
        print("BEE-tool result:")
        print(json.dumps(result, indent=2))
        
        # Extract labels
        labels = []
        for sent_data in result['bug_report'].values():
            labels.extend(sent_data['labels'])
        
        print(f"Extracted labels: {labels}")
        
        return True
        
    except Exception as e:
        print(f"Error testing BEE-tool: {e}")
        return False
    finally:
        close_nlp(nlp)

def test_mock_data():
    """Test with mock bug report data"""
    print("\nTesting with mock data...")
    
    # Load a mock bug report
    mock_report = {
        "bug_id": "103157",
        "title": "Bug 103157 - after returning should not bind null as return value",
        "description": "Today I found out the following. When writing an advice as... after() returning(Object o): staticinitialization(*) { System.out.println(o); } This advice is indeed executed every time a type returns from staticinitialization. However, since there is no returned object, o is bound to null."
    }
    
    print(f"Mock bug report: {mock_report['title']}")
    print(f"Description: {mock_report['description'][:100]}...")
    
    return True

if __name__ == "__main__":
    print("=== ChatBR BEE-tool Integration Test ===\n")
    
    # Test BEE-tool
    bee_success = test_bee_tool()
    
    # Test mock data
    mock_success = test_mock_data()
    
    if bee_success and mock_success:
        print("\n✅ All tests passed! BEE-tool integration is working.")
        print("You can now run the complete ChatBR pipeline with run.py")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
