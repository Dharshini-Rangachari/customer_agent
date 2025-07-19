from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import pickle

def train_intent_classifier(data_path):
    df = pd.read_csv(data_path)
    X = df['query']
    y = df['issue_type']

    # Load SBERT model
    model_name = 'all-MiniLM-L6-v2'
    sbert = SentenceTransformer(model_name)
    embeddings = sbert.encode(X.tolist())

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, y)

    preds = clf.predict(embeddings)
    print("F1 Score:", f1_score(y, preds, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))

    # Save classifier and model name
    pickle.dump(clf, open("intent_model.pkl", "wb"))
    pickle.dump(model_name, open("sbert_model_name.pkl", "wb"))

if __name__ == "__main__":
    train_intent_classifier("data/sample_data.csv")