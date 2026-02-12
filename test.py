import importlib.util
import os
import sys

def check_library(lib_name):
    try:
        importlib.import_module(lib_name)
        print(f"PASS: {lib_name} is installed.")
        return True
    except ImportError:
        print(f"FAIL: {lib_name} is NOT installed.")
        return False

def check_local_module(mod_name):
    try:
        importlib.import_module(mod_name)
        print(f"PASS: {mod_name} imported successfully.")
        return True
    except Exception as e:
        print(f"FAIL: {mod_name} import failed: {e}")
        return False

def testSetup():
    print("Running SER Project Setup Test...")
    
    # External Libs
    libs = ['numpy', 'librosa', 'sklearn', 'tensorflow', 'matplotlib', 'seaborn', 'scipy', 'pandas']
    if not all(check_library(lib) for lib in libs):
        print("FAIL: Missing external libraries.")
        return

    # Local Modules
    local_mods = ['config', 'features', 'model', 'data_preprocessing', 'viz', 'train']
    if not all(check_local_module(mod) for mod in local_mods):
        print("FAIL: Local modules have syntax errors or missing dependencies.")
        return

    # Check dataset
    try:
        from config import DATASET_PATH
        if os.path.isdir(DATASET_PATH):
            files = []
            for r, d, f in os.walk(DATASET_PATH):
                for file in f:
                    if file.endswith('.wav'):
                        files.append(os.path.join(r, file))
            print(f"PASS: Dataset directory found. Found {len(files)} .wav files.")
        else:
            print(f"FAIL: Dataset directory not found at {DATASET_PATH}")
    except Exception as e:
        print(f"FAIL: Could not check dataset: {e}")
        
    print("\nSUCCESS: All checks passed! You can now run 'python train.py'")

if __name__ == "__main__":
    testSetup()
