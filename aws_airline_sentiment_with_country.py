#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
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

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[ ]:


sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

airlines = {
    'Air India': 'https://www.airlinequality.com/airline-reviews/air-india/',
    'British Airways': 'https://www.airlinequality.com/airline-reviews/british-airways/',
    'Qatar Airways': 'https://www.airlinequality.com/airline-reviews/qatar-airways/',
    'Emirates': 'https://www.airlinequality.com/airline-reviews/emirates',
    'Etihad Airways': 'https://www.airlinequality.com/airline-reviews/etihad-airways/'
}

all_reviews = []
keyword_list = []

for airline, base_url in airlines.items():
    print(f"Scraping reviews for {airline}...")
    url = base_url + "page/1/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    review_articles = soup.find_all('article', class_='comp comp_reviews-airline-review')

    for review in review_articles:
        content = review.find('div', class_='text_content').get_text(strip=True) if review.find('div', class_='text_content') else ''
        
        # Extract country
        country_tag = review.find('h3').find('span', class_='review-country') if review.find('h3') else None
        country = country_tag.get_text(strip=True) if country_tag else 'Unknown'
        
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

# Convert to DataFrame
review_df = pd.DataFrame(all_reviews)
print(review_df.head())

# Keyword Frequency Analysis
keyword_counts = Counter(keyword_list)
print("Top Keywords:", keyword_counts.most_common(10))


# In[ ]:


# Store into Amazon RDS (replace with your actual RDS details)
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
    print("❌ Error:", e)

finally:
    conn.close()

