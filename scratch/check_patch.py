import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open("src/models/llama/modeling_llama.py", "r") as f:
    lines = f.readlines()
    print("".join(lines[:20]))
