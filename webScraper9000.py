import os
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from summarizer import Summarizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams

nltk.download('wordnet')
nltk.download('stopwords')


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def remove_empty_cells(df):
    """Clean up the DataFrame to remove empty cells."""
    df = df.dropna().replace('', float("NaN")).dropna().replace(' ', float("NaN")).dropna()
    return df


def save_to_excel(categorized_data, domain_name):
    """Save the categorized data to an Excel file."""
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Border, Side, Alignment

    border = Border(left=Side(style='thin'), 
                    right=Side(style='thin'), 
                    top=Side(style='thin'), 
                    bottom=Side(style='thin'))
    wrap_alignment = Alignment(wrap_text=True)

    today_date = datetime.today().strftime('%m%d%Y')
    file_name = f"{today_date}_{domain_name}_scraped_data.xlsx"
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for category, items in categorized_data.items():
            df = pd.DataFrame(items, columns=[category])
            df_cleaned = remove_empty_cells(df)
            df_cleaned.to_excel(writer, sheet_name=category, index=False)

            # get the workbook and the sheet of interest
            workbook  = writer.book
            worksheet = writer.sheets[category]

            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = border
                    cell.alignment = wrap_alignment

    return file_name


def load_credentials():
    """Load email credentials from environment."""
    load_dotenv()
    email = os.environ['EMAIL']
    email_password = os.environ['EMAIL_PASSWORD']
    return email, email_password


def get_html_content(url, headers):
    """Retrieve the HTML content of a given URL."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.HTTPError as err:
        logging.error(f"Failed to retrieve the web page due to {err}")
        return None
    
    
def categorize_data(soup):
    """Categorize data from the HTML content."""
    categories = {
        "Titles": [header.get_text() for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
        "Content": [para.get_text() for para in soup.find_all('p')],
        "Links": [link.get('href') for link in soup.find_all('a')],
        "Images": [img.get('src') for img in soup.find_all('img')],
        "Meta": [meta.get('content') for meta in soup.find_all('meta') if meta.get('content')]
    }
    return categories

    
def send_email_with_attachment(file_name, email_to, email_from, email_pass):
    """Send email with an attachment."""
    try:
        subject = "Scraped Data"
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = subject

        msg.attach(MIMEText(f"Here is the scraped data from {file_name}", 'plain'))

        with open(file_name, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {file_name}")
            msg.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_from, email_pass)
            server.sendmail(email_from, email_to, msg.as_string())

        logging.info(f"Email sent to {email_to}")

    except Exception as e:
        logging.error(f"Failed to send email due to: {e}")

def analyze_with_ml(categorized_data):
    """Perform ML analysis on the categorized data."""
    analysis_results = {}

    try:
        # Topic Modeling using LDA on Titles
        titles = [title.split() for title in categorized_data["Titles"]]
        dictionary = corpora.Dictionary(titles)
        corpus = [dictionary.doc2bow(text) for text in titles]
        ldamodel = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
        analysis_results["Topics from Titles"] = ldamodel.print_topics(num_words=3)

        # TF-IDF for most important terms in content
        vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english', max_features=10)
        tfidf_matrix = vectorizer.fit_transform(categorized_data["Content"])
        analysis_results["Important Terms from Content"] = vectorizer.get_feature_names_out()

        # Extractive Text Summarization on content
        combined_content = ' '.join(categorized_data["Content"])
        summary = summarize("Title (if any)", combined_content)
        analysis_results["Summarized Content"] = summary

    except Exception as e:
        logging.error(f"Failed to analyze data due to: {e}")

    return analysis_results

def visualize_word_cloud(text):
    wordcloud = WordCloud(stopwords='english', background_color='white', width=800, height=600).generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def connect_to_mongo(cluster_url):
    """Establish connection to the MongoDB cluster."""
    client = MongoClient(cluster_url)
    return client

def save_to_mongo(db_name, collection_name, data, cluster_url):
    """Save data to MongoDB."""
    try:
        client = connect_to_mongo(cluster_url)
        db = client[db_name]
        collection = db[collection_name]
        collection.insert_many(data)
        return True  # return True if data insertion was successful
    except Exception as e:
        logging.error(f"Failed to save to MongoDB due to: {e}")
        return False  # return False if there was an error


def enhanced_preprocessing(text):
    """Perform enhanced text preprocessing."""
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # N-grams (here for bigrams)
    bigrams = list(ngrams(tokens, 2))
    tokens.extend([' '.join(bigram) for bigram in bigrams])

    return ' '.join(tokens)


def test_mongo_connection(cluster_url):
    """Test MongoDB connection."""
    try:
        client = connect_to_mongo(cluster_url)
        # Fetch the server information to verify the connection
        server_info = client.server_info()
        logging.info(f"Successfully connected to MongoDB. Server version: {server_info['version']}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB due to: {e}")
        return False
def collection_exists(client, db_name, collection_name):
    """Check if a collection exists."""
    return collection_name in client[db_name].list_collection_names()

def save_to_mongo(db_name, collection_name, data, cluster_url):
    """Save data to MongoDB."""
    try:
        client = connect_to_mongo(cluster_url)
        
        if collection_exists(client, db_name, collection_name):
            logging.error(f"Collection '{collection_name}' already exists.")
            return False
        
        db = client[db_name]
        collection = db[collection_name]
        collection.insert_many(data)
        return True  # return True if data insertion was successful
    except Exception as e:
        logging.error(f"Failed to save to MongoDB due to: {e}")
        return False  # return False if there was an error
            
def main():
    # Configuration and Initialization
    url = input("Please enter the URL you want to scrape: ")
    if not url.startswith('http'):
        logging.error("Please enter a valid URL.")
        return
    email_to = 'robert.taku.hartley@gmail.com'
    email, email_password = load_credentials()
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }
    
    # Web Scraping
    html_content = get_html_content(url, HEADERS)
    if not html_content:
        logging.error("Could not retrieve the HTML content.")
        return
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Data Categorization and Saving to Excel
    categorized_data = categorize_data(soup)
    domain_name = urlparse(url).netloc.split('.')[-2]
    file_name = save_to_excel(categorized_data, domain_name)
    
    # Sending Email
    send_email_with_attachment(file_name, email_to, email, email_password)
    
    # Visualization
    combined_content = ' '.join(categorized_data["Content"])
    visualize_word_cloud(combined_content)
    
    # ML Analysis
    analysis_results = analyze_with_ml(categorized_data)
    for key, value in analysis_results.items():
        logging.info(f"{key}: {value}")
    
     # MongoDB Configuration
    MONGO_CLUSTER_URL = os.getenv("MONGO_CLUSTER_URL")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    # Get the MongoDB collection name from user
    MONGO_COLLECTION_NAME = input("Please enter the MongoDB collection name: ")
     # Test MongoDB Connection
    if not test_mongo_connection(MONGO_CLUSTER_URL):
        logging.error("Exiting the script as MongoDB connection failed.")
        return
    
    # Text Preprocessing
    processed_content = enhanced_preprocessing(combined_content)
    
    # Saving to MongoDB
    data_to_save = {
        "date": datetime.today().strftime('%Y-%m-%d'),
        "url": url,
        "content": processed_content
    }
    save_successful = save_to_mongo(MONGO_DB_NAME, MONGO_COLLECTION_NAME, [data_to_save], MONGO_CLUSTER_URL)


    if save_successful:
        logging.info("Data successfully saved to MongoDB.")
    else:
        logging.error("Failed to save data to MongoDB.")
if __name__ == '__main__':
    main()
