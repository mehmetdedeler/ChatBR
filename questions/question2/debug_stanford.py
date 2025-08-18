#!/usr/bin/env python3
"""
Debug Stanford CoreNLP annotation issue
"""

import json
from stanfordcorenlp import StanfordCoreNLP

def test_stanford_annotation():
    """Test Stanford CoreNLP annotation"""
    print("=== Testing Stanford CoreNLP Annotation ===\n")
    
    # Initialize Stanford CoreNLP
    try:
        nlp = StanfordCoreNLP('http://localhost', port=9000)
        print("✅ Connected to Stanford CoreNLP")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Test sentence
    test_sentence = "The application crashes when clicking the button."
    print(f"Testing sentence: '{test_sentence}'")
    
    try:
        # Get annotation
        print("\nGetting annotation...")
        ann = nlp.annotate(test_sentence, properties={
            'annotators': 'tokenize,ssplit,lemma,pos,ner',
            'outputFormat': 'json',
        })
        
        print(f"Raw annotation length: {len(ann)}")
        print(f"Raw annotation preview: {ann[:200]}...")
        
        # Try to parse JSON
        print("\nParsing JSON...")
        parsed = json.loads(ann)
        
        print("✅ JSON parsed successfully!")
        print(f"Number of sentences: {len(parsed['sentences'])}")
        
        if len(parsed['sentences']) > 0:
            tokens = parsed['sentences'][0]['tokens']
            print(f"Number of tokens: {len(tokens)}")
            
            # Show first few tokens
            print("\nFirst 5 tokens:")
            for i, token in enumerate(tokens[:5]):
                print(f"  {i}: word='{token['word']}', lemma='{token['lemma']}', pos='{token['pos']}'")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response: {repr(ann)}")
    except Exception as e:
        print(f"❌ Other error: {e}")
    
    finally:
        nlp.close()
        print("\n✅ Stanford CoreNLP connection closed")

if __name__ == "__main__":
    test_stanford_annotation()
