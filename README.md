# Create a project directory and navigate into it (optional, if you're not already there)
mkdir my_streamlit_app_project
cd my_streamlit_app_project

# Create a virtual environment
python -m venv myenv

# Activate the virtual environment and install dependencies
# For Windows Command Prompt, use: .\myenv\Scripts\activate.bat && pip install -r requirements.txt

# For Windows PowerShell, use: .\myenv\Scripts\Activate.ps1; pip install -r requirements.txt
source myenv/bin/activate && pip install -r requirements.txt

# Run the Streamlit app
streamlit run App.py

