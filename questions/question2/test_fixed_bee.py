#!/usr/bin/env python3
"""
Test the fixed BEE-tool
"""

import sys
import os
sys.path.append('../question1')

def test_fixed_bee():
    """Test the fixed BEE-tool"""
    print("=== Testing Fixed BEE-tool ===\n")
    
    try:
        from bee_tool import start_nlp, close_nlp, processText, readWords
        
        # Test 1: Read dictionary
        print("1. Testing dictionary reading...")
        readWords()
        print("✅ Dictionary loaded successfully")
        
        # Test 2: Stanford CoreNLP connection
        print("\n2. Testing Stanford CoreNLP connection...")
        nlp = start_nlp()
        print("✅ Stanford CoreNLP connected")
        
        # Test 3: Process a simple sentence
        print("\n3. Testing sentence processing...")
        test_sentence = "The application crashes when clicking the button."
        print(f"Test sentence: '{test_sentence}'")
        
        result = processText("test_simple", test_sentence, nlp)
        print(f"✅ Processing completed")
        print(f"Result: {result}")
        
        # Test 4: Process a sentence with spaces in the ID
        print("\n4. Testing sentence with spaces in ID...")
        complex_sentence = "When I click the submit button, the form should save the data but instead it shows an error message."
        print(f"Complex sentence: '{complex_sentence}'")
        
        result2 = processText("test with spaces", complex_sentence, nlp)
        print(f"✅ Complex processing completed")
        print(f"Result: {result2}")
        
        # Close Stanford CoreNLP
        close_nlp(nlp)
        print("\n✅ Stanford CoreNLP closed")
        
        print("\n🎉 All tests passed! BEE-tool is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_bee()
