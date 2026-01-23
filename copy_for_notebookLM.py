"""
Simple utility to copy Python files to .txt format for NotebookLM upload.
NotebookLM accepts .txt files but not .py files.
"""
import shutil
import os

# Define files to copy (Source -> Destination)
file_map = {
    'Level_Matcher.py': 'Level_Matcher.txt',
    'Feature_Engineer.py': 'Feature_Engineer.txt',
    'Dataset_Parser.py': 'Dataset_Parser.txt',
    'Combined_Visualizer.py': 'Combined_Visualizer.txt',
    'scripts/hyperparameter_tuning/Hyperparameter_Tuner.py': 'Hyperparameter_Tuner.txt'
}

print("[INFO] Converting Python scripts to Text for NotebookLM...")

for src, dst in file_map.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  - {src} → {dst}")
    else:
        print(f"  [WARNING] Source file not found: {src}")

print("[INFO] Conversion complete.")
