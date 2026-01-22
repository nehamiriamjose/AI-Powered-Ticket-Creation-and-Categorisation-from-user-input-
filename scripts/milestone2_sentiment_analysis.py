import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only runs once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ---------------- TESTING ----------------
if __name__ == "__main__":
    sample_tickets = [
        "Please reset my password when possible",
        "My system is down and this is extremely frustrating",
        "VPN is not working"
    ]

    for ticket in sample_tickets:
        sentiment = analyze_sentiment(ticket)
        print(f"Ticket: {ticket}")
        print(f"Sentiment: {sentiment}\n")
