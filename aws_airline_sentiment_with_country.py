import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import pymysql

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Airline URLs
airlines = {
    'Air India': 'https://www.airlinequality.com/airline-reviews/air-india/',
    'British Airways': 'https://www.airlinequality.com/airline-reviews/british-airways/',
    'Qatar Airways': 'https://www.airlinequality.com/airline-reviews/qatar-airways/',
    'Emirates': 'https://www.airlinequality.com/airline-reviews/emirates/',
    'Etihad Airways': 'https://www.airlinequality.com/airline-reviews/etihad-airways/'
}

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
driver = webdriver.Chrome(options=chrome_options)

# Scrape reviews
all_reviews = []
keyword_list = []

# NLTK downloads
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

for airline, base_url in airlines.items():
    print(f"Scraping reviews for {airline}...")
    for page in range(1, 4):  # Scrape 3 pages
        url = f"{base_url}page/{page}/"
        retries = 3  # Retry logic
        while retries > 0:
            try:
                driver.get(url)
                time.sleep(random.randint(3, 6))  # Randomized delay to prevent bot detection
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                review_articles = soup.find_all('article', class_='comp comp_media-review-rated list-item media position-content')

                if not review_articles:
                    print(f"❌ No reviews found on page {page} for {airline}. Moving to next page.")
                    break

                print(f"✅ Found {len(review_articles)} reviews on page {page} for {airline}")
                for review in review_articles:
                    content_div = review.find('div', class_='text_content')
                    content = content_div.get_text(strip=True) if content_div else 'No Content Found'

                    country_tag = review.find('h3', class_='text_sub_header userStatusWrapper')
                    country = 'Unknown'
                    if country_tag and '(' in country_tag.text:
                        country = country_tag.text.split('(')[-1].replace(')', '').strip()

                    # NLP Preprocessing
                    tokens = word_tokenize(content.lower())
                    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
                    pos_tags = nltk.pos_tag(tokens)

                    # Collect nouns/adjectives as keywords
                    keywords = [word for word, pos in pos_tags if pos in ('NN', 'NNS', 'JJ')]
                    keyword_list.extend(keywords)

                    sentiment = sia.polarity_scores(content)

                    all_reviews.append({
                        'airline': airline,
                        'review_date': datetime.utcnow().strftime('%Y-%m-%d'),
                        'review_text': content,
                        'processed_text': ' '.join(tokens),
                        'country': country,
                        'sentiment_score': sentiment['compound']
                    })
                break  # Break out of retry loop if successful
            except Exception as e:
                retries -= 1
                print(f"❌ Error on page {page} for {airline}: {str(e)}")
                if retries > 0:
                    print("Retrying...")
                    time.sleep(5 + random.randint(0, 5))  # Random delay before retry
                else:
                    print("Max retries reached, moving to next page.")

driver.quit()

# Convert to DataFrame
import pandas as pd
review_df = pd.DataFrame(all_reviews)
print(f"Total Reviews Scraped: {len(review_df)}")
print(review_df.head())

# Keyword Frequency
from collections import Counter
keyword_counts = Counter(keyword_list)
print("Top Keywords:", keyword_counts.most_common(10))

# Store into Amazon RDS MySQL
try:
    conn = pymysql.connect(
        host='airlinereview-db.c8xg22su41px.us-east-1.rds.amazonaws.com',
        user='admin',
        password='airline123',
        database='airline_reviews'
    )
    cursor = conn.cursor()

    for _, row in review_df.iterrows():
        sql = """
            INSERT INTO reviews (airline, review_date, country, sentiment_score, review_text, processed_text)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (row['airline'], row['review_date'], row['country'], row['sentiment_score'], row['review_text'], row['processed_text']))

    conn.commit()
    print("✅ Data successfully inserted into RDS")

except Exception as e:
    print("❌ Error while inserting into RDS:", e)

finally:
    if 'conn' in locals():
        conn.close()
