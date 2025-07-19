# AI-Based Customer Support Agent (SBERT + Q-Learning)

This project implements an **intelligent customer support agent** using **Sentence-BERT (SBERT)** for natural language understanding and **Q-learning** for decision-making. The goal is to automate customer query classification and select optimal resolution strategies based on issue type, urgency, and feedback.

## Features

- **NLP-based Intent Classifier**: Uses SBERT to classify customer queries as billing, technical, etc.
- **Q-Learning Agent**: Learns optimal support actions based on issue type and urgency.
- **Simulation Engine**: Simulates customer interaction episodes to train the agent.
- **Streamlit UI**: Provides an interactive interface to test the agent.


## 🧪 Technologies Used

- `sentence-transformers` (SBERT)
- `scikit-learn` (Logistic Regression)
- `matplotlib` (reward visualization)
- `streamlit` (user interface)
- `pandas`, `numpy`


## 🚀 How to Run

### 1. Setup

```bash
python -m venv venv
source venv/Scripts/activate  # On Git Bash (Windows)
pip install -r requirements.txt

2. Train Intent Classifier
python intent_classifier.py

3. Train Q-Learning Agent
python simulate_agent.py

Launch Streamlit UI
streamlit run streamlit_ui.py
