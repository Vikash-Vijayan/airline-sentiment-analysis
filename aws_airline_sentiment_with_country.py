import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import random
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from requests.exceptions import RequestException

# Setup NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to scrape reviews
def scrape_reviews(airline, num_pages=5):
    reviews = []
    
    for page in range(1, num_pages + 1):
        url = f"https://www.airlinequality.com/airline-reviews/{airline}/page/{page}/"
        
        try:
            # Send request and get the page content
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse the page content
            soup = BeautifulSoup(response.content, 'html.parser')
            review_elements = soup.find_all('div', class_='text_content')
            
            # Extract reviews
            for review in review_elements:
                review_text = review.get_text(strip=True)
                reviews.append(review_text)
        
        except RequestException as e:
            logging.error(f"Error while scraping {airline} on page {page}: {e}")
            time.sleep(2 + random.randint(0, 3))  # Backoff time for retry

    logging.info(f"Scraped {len(reviews)} reviews for {airline}")
    return reviews

# Sentiment analysis function
def analyze_sentiment(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for review in reviews:
        score = sia.polarity_scores(review)
        sentiment_scores.append(score['compound'])
    
    return sentiment_scores

# Function to save data to AWS RDS (MySQL/PostgreSQL)
def save_to_rds(airline, reviews, sentiment_scores, db_url):
    # Create a DataFrame
    df = pd.DataFrame({
        'airline': [airline] * len(reviews),
        'review': reviews,
        'sentiment_score': sentiment_scores
    })
    
    try:
        # Create database connection
        engine = create_engine(db_url)
        
        # Save the DataFrame to the database
        df.to_sql('airline_reviews', con=engine, if_exists='append', index=False)
        logging.info(f"Successfully saved {len(reviews)} reviews to RDS for {airline}")
    
    except SQLAlchemyError as e:
        logging.error(f"Error saving data to RDS: {e}")

# Main function to run the scraping, sentiment analysis, and saving to database
def main():
    airline = "air-india"  # Change to desired airline
    db_url = "mysql+pymysql://username:password@host/db_name"  # Replace with your RDS connection details
    
    reviews = scrape_reviews(airline, num_pages=5)
    
    if reviews:
        sentiment_scores = analyze_sentiment(reviews)
        save_to_rds(airline, reviews, sentiment_scores, db_url)
    else:
        logging.error(f"No reviews found for {airline}")

if __name__ == "__main__":
    main()
