import sys
import os

# Get the current notebook's directory
current_dir = os.getcwd()

# Construct the path to the src directory
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))

# Add src to the system path
sys.path.append(src_dir)
