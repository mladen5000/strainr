import sys
import os

print(f"CONTEST_PY: sys.executable: {sys.executable}")
print(f"CONTEST_PY: sys.path: {sys.path}")
print(f"CONTEST_PY: os.environ['PATH']: {os.environ.get('PATH')}")
print(f"CONTEST_PY: os.environ.get('VIRTUAL_ENV'): {os.environ.get('VIRTUAL_ENV')}")
