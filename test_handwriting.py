import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Seamlessly execute the full Streamlit Lab application
# Ensures Streamlit Cloud succeeds whether configured with app.py or test_handwriting.py
import app
