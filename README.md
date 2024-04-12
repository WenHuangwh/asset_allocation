# Asset Prediction Model

This project combines GARCH and Naive Bayesian Network models to predict asset returns, volatility, and optimize asset allocation.

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Setting Up a Virtual Environment

It's good practice to use a virtual environment for your Python projects. To set one up and activate it, follow these steps:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate
# On macOS and Linux
source venv/bin/activate

# Installing Dependencies
# With your virtual environment activated, install the required packages using pip:
pip install -r requirements.txt

# Deactivating Virtual Environment:
# Once you're done working in the virtual environment and want to return to the global Python environment, you can deactivate it:
deactivate