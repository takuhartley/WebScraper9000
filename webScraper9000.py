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
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        "Links": [link.get('href') for link in soup.find_all('a')]
    }
    return categories


def remove_empty_cells(df):
    """Clean up the DataFrame to remove empty cells."""
    df = df.dropna().replace('', float("NaN")).dropna().replace(' ', float("NaN")).dropna()
    return df


def save_to_excel(categorized_data, domain_name):
    """Save the categorized data to an Excel file."""
    today_date = datetime.today().strftime('%m%d%Y')
    file_name = f"{today_date}_{domain_name}_scraped_data.xlsx"
    with pd.ExcelWriter(file_name) as writer:
        for category, items in categorized_data.items():
            df = pd.DataFrame(items, columns=[category])
            df_cleaned = remove_empty_cells(df)
            df_cleaned.to_excel(writer, sheet_name=category, index=False)
    return file_name

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
        model = Summarizer()
        combined_content = ' '.join(categorized_data["Content"])
        summary = model.predict(combined_content)
        analysis_results["Summarized Content"] = summary

    except Exception as e:
        logging.error(f"Failed to analyze data due to: {e}")

    return analysis_results

def main():
    url = 'https://finance.yahoo.com'
    email_to = 'robert.taku.hartley@gmail.com'
    email, email_password = load_credentials()

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    html_content = get_html_content(url, HEADERS)
    if not html_content:
        logging.error("Could not retrieve the HTML content.")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    categorized_data = categorize_data(soup)
    domain_name = urlparse(url).netloc.split('.')[-2]
    file_name = save_to_excel(categorized_data, domain_name)
    send_email_with_attachment(file_name, email_to, email, email_password)

    analysis_results = analyze_with_ml(categorized_data)
    for key, value in analysis_results.items():
        logging.info(f"{key}: {value}")


if __name__ == '__main__':
    main()
