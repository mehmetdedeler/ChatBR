#!/usr/bin/env python3
"""
Script to start Stanford CoreNLP server
"""

import subprocess
import time
import os
import sys

def start_stanford_corenlp():
    """Start Stanford CoreNLP server"""
    print("Starting Stanford CoreNLP server...")
    
    # Path to Stanford CoreNLP
    corenlp_path = "../../utils/stanford-corenlp-full-2018-01-31"
    
    if not os.path.exists(corenlp_path):
        print(f"Error: Stanford CoreNLP not found at {corenlp_path}")
        return False
    
    # Command to start Stanford CoreNLP
    cmd = [
        "java", "-mx4g", "-cp", "*", 
        "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", 
        "-port", "9000", "-timeout", "15000"
    ]
    
    try:
        # Change to Stanford CoreNLP directory
        os.chdir(corenlp_path)
        
        # Start the server
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        print("Waiting for server to start...")
        time.sleep(10)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Stanford CoreNLP server started successfully!")
            print("Server is running on port 9000")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Stanford CoreNLP: {e}")
        return False

if __name__ == "__main__":
    success = start_stanford_corenlp()
    if success:
        print("\nYou can now run the ChatBR implementation!")
        print("Run: python run_chatbr.py")
    else:
        print("\nFailed to start Stanford CoreNLP. Please check the installation.")
