import pandas as pd
from q_learning_agent import QLearningAgent
import pickle
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Define environment
actions = ['assign_bot', 'escalate_L2', 'suggest_article']
states = ['billing_low', 'billing_high', 'tech_low', 'tech_high']

def get_reward(action, feedback):
    if feedback == "positive":
        return 10 if action != 'escalate_L2' else 5
    else:
        return -10 if action == 'suggest_article' else -5

def simulate():
    # Load data and models
    df = pd.read_csv("data/sample_data.csv")
    clf = pickle.load(open("intent_model.pkl", "rb"))
    model_name = pickle.load(open("sbert_model_name.pkl", "rb"))
    sbert = SentenceTransformer(model_name)

    agent = QLearningAgent(states, actions)
    episode_rewards = []

    for ep in range(1000):
        total_reward = 0
        for _, row in df.iterrows():
            query = row['query']
            urgency = row['urgency']
            feedback = row['feedback']

            embedding = sbert.encode([query])
            issue = clf.predict(embedding)[0]
            state = f"{issue}_{urgency}"

            action = agent.choose_action(state)
            reward = get_reward(action, feedback)
            next_state = state

            agent.update(state, action, reward, next_state)
            total_reward += reward

        episode_rewards.append(total_reward)
        if ep % 100 == 0:
            print(f"Episode {ep} - Total Reward: {total_reward}")

    agent.save()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Agent Performance")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate()