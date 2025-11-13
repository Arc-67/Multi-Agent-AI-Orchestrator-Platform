# Clone Repo
git clone https://github.com/Arc-67/chatbot_agent.git
cd chatbot_agent

# Create Virtual env
python -m venv .venv

# Activate Virtual env (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate Virtual env (CMD)
.\.venv\Scripts\activate.bat

# Install Requirements
pip install -r requirements.txt

# Create an .env and edit it to add API keys
OPENAI_API_KEY=""
PINECONE_API_KEY = ""

# Run chatbot app locally:
uvicorn main:app --reload