#!/usr/bin/env python3
import subprocess
import sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
    cwd=r'c:\Users\kjuetten\Documents\GitHub\Beacon-Low-Training-Pi-Camera-System'
)
sys.exit(result.returncode)
