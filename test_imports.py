#!/usr/bin/env python3
import sys
import os

# Add workspace root to path
root = r'c:\Users\kjuetten\Documents\GitHub\Beacon-Low-Training-Pi-Camera-System'
os.chdir(root)
if root not in sys.path:
    sys.path.insert(0, root)

# Try importing the main modules
modules_to_test = [
    'inspection_system.app.camera_interface',
    'inspection_system.app.frame_acquisition',
    'inspection_system.app.anomaly_detection_utils',
    'inspection_system.app.inspection_pipeline',
    'inspection_system.app.capture_test',
    'inspection_system.app.replay_inspection',
    'inspection_system.app.replay_summary',
]

print("Testing module imports...\n")
for mod in modules_to_test:
    try:
        __import__(mod)
        print(f'OK: {mod}')
    except Exception as e:
        print(f'FAIL: {mod}')
        print(f'  Error: {str(e)[:100]}')
        import traceback
        traceback.print_exc()
        
print("\nDone!")
