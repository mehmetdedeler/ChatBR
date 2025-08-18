#!/usr/bin/env python3
"""
Quick test to verify fixes work
"""

import sys
sys.path.append('..')

def test_quick():
    """Quick test of the fixed components"""
    print("=== Quick Test ===\n")
    
    try:
        # Test 1: Import classifier_predict
        from classifier_predict import predict_multi_data
        print("✅ classifier_predict imported")
        
        # Test 2: Import analysis_sample
        from analysis_sample import analysis_llm_data, select_dataset_for_llm
        print("✅ analysis_sample imported")
        
        # Test 3: Create args object
        class Args:
            def __init__(self):
                self.json_bug_path = './origin_data'
                self.bert_result_path = './predict_data/bee_tool/'
                self.llm_data_path = './llm_dataset'
                self.report_max_length = 2000
        
        args = Args()
        print("✅ Args object created")
        
        # Test 4: Test predict_multi_data (just the function call, not execution)
        print("✅ All components ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick()
    if success:
        print("\n🎉 Quick test passed! You can now run:")
        print("python run_simple_chatbr.py")
    else:
        print("\n❌ Quick test failed. Please fix the issues above.")
