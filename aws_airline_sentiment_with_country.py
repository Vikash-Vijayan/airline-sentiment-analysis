import requests
from bs4 import BeautifulSoup
import pandas as pd
import mysql.connector
import re
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# ✅ Amazon RDS Database Configuration
rds_host = 'database-2.c8xg22su41px.us-east-1.rds.amazonaws.com'
rds_user = 'admin'
rds_password = 'admin123'
rds_database = 'airline_reviews'

# ✅ Establish RDS connection
def rds_connection():
    return mysql.connector.connect(
        host=rds_host,
        user=rds_user,
        password=rds_password,
        database=rds_database
    )

# ✅ Airline URLs
airlines = {
    'Air India': 'https://www.airlinequality.com/airline-reviews/air-india/',
    'British Airways': 'https://www.airlinequality.com/airline-reviews/british-airways/',
    'Qatar Airways': 'https://www.airlinequality.com/airline-reviews/qatar-airways/',
    'Emirates': 'https://www.airlinequality.com/airline-reviews/emirates/',
    'Etihad Airways': 'https://www.airlinequality.com/airline-reviews/etihad-airways/'
}

# ✅ Scrape reviews function
def scrape_airline_reviews(airline, url, pages=3):
    reviews_list = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    for page in range(1, pages + 1):
        print(f"Scraping {airline} - Page {page}")
        response = requests.get(f"{url}page/{page}/", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"❌ Failed to fetch {airline} - Page {page}")
            continue

        soup = BeautifulSoup(response.text, "html5lib")
        review_divs = soup.find_all("div", class_="text_content")
        print(f"✅ Found {len(review_divs)} reviews on page {page}")

        h3_dates = soup.find_all('time')
        h3_country = soup.find_all(class_="text_sub_header userStatusWrapper")

        country_list = []
        for tag in h3_country:
            text = tag.get_text()
            country_match = re.search(r'\((.*?)\)', text)
            country_list.append(country_match.group(1) if country_match else "Unknown")

        dates_list = [h3.text.strip() for h3 in h3_dates]

        for idx, div in enumerate(review_divs):
            review_text = div.get_text(strip=True)
            country_text = country_list[idx] if idx < len(country_list) else "Unknown"
            date_text = dates_list[idx] if idx < len(dates_list) else "Unknown"

            reviews_list.append({
                "Airline": airline,
                "Review_Date": date_text,
                "Review_Text": review_text,
                "Country": country_text
            })
        time.sleep(1)
    return reviews_list

# ✅ Perform Sentiment Analysis + Add Rating
def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    labels = []
    ratings = []

    for text in df['Review_Text']:
        sentiment_score = sid.polarity_scores(text)['compound']
        sentiments.append(sentiment_score)

        if sentiment_score > 0.05:
            label = 'Positive'
            rating = 5
        elif sentiment_score < -0.05:
            label = 'Negative'
            rating = 1
        else:
            label = 'Neutral'
            rating = 3

        labels.append(label)
        ratings.append(rating)

    df['Sentiment_Score'] = sentiments
    df['Sentiment_Label'] = labels
    df['Rating'] = ratings
    return df

# ✅ Insert Data into RDS MySQL if NOT EXISTS
def insert_to_rds(df):
    conn = rds_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS airline_reviews (
            id INT AUTO_INCREMENT PRIMARY KEY,
            airline_name VARCHAR(255),
            review_date VARCHAR(255),
            review_text TEXT,
            country VARCHAR(255),
            sentiment_score FLOAT,
            sentiment_label VARCHAR(50),
            rating INT
        )
    ''')

    inserted_count = 0

    for _, row in df.iterrows():
        # ✅ Check if this review already exists
        cursor.execute('''
            SELECT COUNT(*) FROM airline_reviews
            WHERE airline_name = %s AND review_date = %s AND review_text = %s
        ''', (row['Airline'], row['Review_Date'], row['Review_Text']))
        result = cursor.fetchone()

        if result[0] == 0:
            # ✅ Insert new review with rating
            cursor.execute('''
                INSERT INTO airline_reviews 
                (airline_name, review_date, review_text, country, sentiment_score, sentiment_label, rating)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (row['Airline'], row['Review_Date'], row['Review_Text'], row['Country'],
                  row['Sentiment_Score'], row['Sentiment_Label'], row['Rating']))
            inserted_count += 1
        else:
            print(f"⚠️ Duplicate found. Skipping review for {row['Airline']} on {row['Review_Date']}")

    conn.commit()
    print(f"✅ {inserted_count} new reviews inserted successfully.")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    all_reviews = []
    for airline, url in airlines.items():
        airline_reviews = scrape_airline_reviews(airline, url, pages=10)
        all_reviews.extend(airline_reviews)

    # ✅ Convert to DataFrame
    df_reviews = pd.DataFrame(all_reviews)
    print("✅ Reviews scraped. Performing sentiment analysis...")

    if df_reviews.empty:
        print("❌ No reviews scraped. Exiting...")
        exit()

    # ✅ Perform Sentiment Analysis + Add Rating
    df_reviews = analyze_sentiment(df_reviews)
    print(df_reviews.head())

    # ✅ Insert into RDS only if not duplicate
    insert_to_rds(df_reviews)
    print("✅ Process completed. Data inserted without duplicates!")
