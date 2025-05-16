import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Extended training data with Disaster sentiment
data = {
    "Review": [
        "great", "excellent", "awesome", "good", "fantastic", 
        "bad", "boring", "worst", "terrible", "poor",
        "okay", "average", "not bad", "fine", "meh",
        "disaster", "horrible", "awful", "catastrophe", "dreadful"
    ],
    "Sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
        "Disaster", "Disaster", "Disaster", "Disaster", "Disaster"
    ]
}

df = pd.DataFrame(data)

le = LabelEncoder()
y = le.fit_transform(df["Sentiment"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Review"])

model = RandomForestClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, le), f)

print("Model trained and saved as model.pkl")
