
# Make sure you're in the directory where you want your project to live,
# or uncomment the lines below to create and enter a new project folder!
# mkdir my_awesome_streamlit_app
# cd my_awesome_streamlit_app

echo "🚀 Setting up your Streamlit environment..."

# Create a clean virtual environment to keep things tidy
python -m venv myenv

# Detect your operating system and activate the environment, then install dependencies
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash / MinGW)
    ./myenv/Scripts/activate && pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "⚠️  Dependency installation failed. Trying PowerShell activation for Windows."
        # Fallback for PowerShell (requires manual paste of activate.ps1 part)
        echo "Please manually run: .\\myenv\\Scripts\\Activate.ps1; pip install -r requirements.txt"
        read -p "Press Enter after running the above command..."
    fi
elif [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS / Linux
    source myenv/bin/activate && pip install -r requirements.txt
else
    echo "❗ Unable to detect OS for activation. Please activate your virtual environment manually:"
    echo "   - Windows (CMD): .\\myenv\\Scripts\\activate.bat"
    echo "   - Windows (PowerShell): .\\myenv\\Scripts\\Activate.ps1"
    echo "   - macOS/Linux: source myenv/bin/activate"
    echo "Then run: pip install -r requirements.txt"
    read -p "Press Enter once activated and dependencies are installed..."
fi

echo "✅ Environment ready! Launching your Streamlit app..."

# Fire up your Streamlit application!
streamlit run App.py

# --- End of Quickstart ---
