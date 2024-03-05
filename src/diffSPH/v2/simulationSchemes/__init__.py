import os
import glob

# Get the current directory
current_dir = os.path.dirname(__file__)

# Get all the Python files in the current directory
py_files = glob.glob(os.path.join(current_dir, "*.py"))

# Import all the Python files as modules
for py_file in py_files:
    module_name = os.path.basename(py_file)[:-3]  # Remove the ".py" extension
    __import__(module_name, globals(), locals(), level=1)

del os, glob, current_dir, py_files, py_file, module_name