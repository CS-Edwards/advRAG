#!/bin/bash

# Create a virtual environment named myenv
python -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install requirements
pip install -r requirements.txt

# Print
echo "Setup completed. Virtual environment 'myenv' is created and requirements are installed."
