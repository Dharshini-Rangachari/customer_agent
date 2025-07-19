import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from q_learning_agent import QLearningAgent

model = pickle.load(open("intent_model.pkl", "rb"))
model_name = pickle.load(open("sbert_model_name.pkl", "rb"))
sbert = SentenceTransformer(model_name)
agent = QLearningAgent(
    states=['billing_low', 'billing_high', 'tech_low', 'tech_high'],
    actions=['assign_bot', 'escalate_L2', 'suggest_article']
)
agent.load("q_table.pkl")

def get_reward(action, feedback):
    if feedback == "positive":
        return 10 if action != 'escalate_L2' else 5
    else:
        return -10 if action == 'suggest_article' else -5

st.title("AI Customer Support Agent")

query = st.text_input("Enter customer query:")
urgency = st.selectbox("Select urgency:", ["low", "high"])
feedback = st.selectbox("Expected feedback (for simulating reward):", ["positive", "negative"])

if st.button("Submit"):
    issue = model.predict([sbert.encode(query)])[0]
    state = f"{issue}_{urgency}"
    action = agent.choose_action(state)
    reward = get_reward(action, feedback)

    st.markdown(f"**Predicted Issue Type:** `{issue}`")
    st.markdown(f"**Chosen Action by Agent:** `{action}`")
    st.markdown(f"**Simulated Reward:** `{reward}`")