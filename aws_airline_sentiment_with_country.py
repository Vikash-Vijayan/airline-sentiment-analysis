from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import pymysql
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import time

# NLTK downloads
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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

all_reviews = []
keyword_list = []

# ✅ Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ✅ Use system-installed Chrome
driver = webdriver.Chrome(options=chrome_options)

for airline, base_url in airlines.items():
    print(f"Scraping reviews for {airline}...")
    for page in range(1, 4):  # Scrape 3 pages
        url = f"{base_url}page/{page}/"
        driver.get(url)
        
        # Wait for the page content to load (adjust the selector as needed)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'comp_media-review-rated')))
        
        time.sleep(2)  # Let the page load fully

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        review_articles = soup.find_all('article', class_='comp comp_media-review-rated list-item media position-content')
        print(f"✅ Found {len(review_articles)} reviews on page {page} for {airline}")

        for review in review_articles:
            content_div = review.find('div', class_='text_content')
            content = content_div.get_text(strip=True) if content_div else 'No Content Found'

            country_tag = review.find('h3', class_='text_sub_header userStatusWrapper')
            country = 'Unknown'
            if country_tag:
                country = country_tag.text.strip()

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

driver.quit()

# ✅ Convert to DataFrame
review_df = pd.DataFrame(all_reviews)
print(review_df.head())
print(f"Total Reviews Scraped: {len(review_df)}")

# ✅ Keyword Frequency
keyword_counts = Counter(keyword_list)
print("Top Keywords:", keyword_counts.most_common(10))

# ✅ Store into Amazon RDS MySQL
try:
    conn = pymysql.connect(
        host='airlinereview-db.c8xg22su41px.us-east-1.rds.amazonaws.com',
        user='admin',
        password='airline123',
        database='airline_reviews'
    )
    cursor = conn.cursor()

    # Batch Insert
    rows_to_insert = []
    for _, row in review_df.iterrows():
        rows_to_insert.append((row['airline'], row['review_date'], row['country'], row['sentiment_score'], row['review_text'], row['processed_text']))

    sql = """
        INSERT INTO reviews (airline, review_date, country, sentiment_score, review_text, processed_text)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(sql, rows_to_insert)
    conn.commit()
    print("✅ Data successfully inserted into RDS")

except Exception as e:
    print("❌ Error while inserting into RDS:", e)

finally:
    if 'conn' in locals():
        conn.close()
