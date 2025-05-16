from flask import Flask, render_template, request
import pandas as pd
import os
import pickle

app = Flask(__name__)

# Load model, vectorizer, and label encoder
with open("model.pkl", "rb") as f:
    model, vectorizer, le = pickle.load(f)

def predict_sentiment(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return le.inverse_transform([pred])[0]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    past_reviews = []

    if request.method == "POST":
        movie = request.form["movie"]
        review = request.form["review"]
        prediction = predict_sentiment(review)

        new_entry = pd.DataFrame([[movie, review, prediction]], columns=["Movie", "Review", "Sentiment"])

        if os.path.exists("saved_reviews.csv"):
            existing = pd.read_csv("saved_reviews.csv")
            updated = pd.concat([new_entry, existing], ignore_index=True)
        else:
            updated = new_entry

        updated.to_csv("saved_reviews.csv", index=False)
        past_reviews = updated.to_dict(orient="records")

    elif os.path.exists("saved_reviews.csv"):
        existing = pd.read_csv("saved_reviews.csv")
        past_reviews = existing.to_dict(orient="records")

    return render_template("index.html", prediction=prediction, past_reviews=past_reviews)

@app.route("/movie_reviews", methods=["GET", "POST"])
def movie_reviews():
    reviews = []
    searched = None
    summary = None

    if request.method == "POST":
        searched = request.form["movie"]
        if os.path.exists("saved_reviews.csv"):
            df = pd.read_csv("saved_reviews.csv")
            filtered = df[df["Movie"].str.lower() == searched.lower()]
            reviews = filtered.to_dict(orient="records")
            if not filtered.empty:
                counts = filtered["Sentiment"].value_counts()
                summary = counts.idxmax()

    return render_template("review.html", reviews=reviews, searched=searched, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
