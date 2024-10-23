from sec_edgar_downloader import Downloader
import bs4
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from datetime import datetime, timedelta

import os



class SECAnalyzer:
    def __init__(self, company_name, email_address):
        self.downloader = Downloader(company_name, email_address)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
    def get_filing_path(self, ticker, filing_type):
    #Get path of downloaded filing
        try:
            base_path = os.path.join("sec-edgar-filings", ticker, filing_type)
            print(f"Looking for files in: {base_path}")
            
            # First, we need to get the CIK directory
            if os.path.exists(base_path):
                # List all directories (should be the CIK directory)
                subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if subdirs:
                    # Use the first (and should be only) subdir
                    cik_dir = subdirs[0]
                    full_path = os.path.join(base_path, cik_dir)
                    print(f"Found CIK directory: {cik_dir}")
                    
                    # Now look for our files in this directory
                    if os.path.exists(os.path.join(full_path, "primary-document.html")):
                        return os.path.join(full_path, "primary-document.html")
                    elif os.path.exists(os.path.join(full_path, "full-submission.txt")):
                        return os.path.join(full_path, "full-submission.txt")
            
            print("Available files and directories:")
            for root, dirs, files in os.walk(base_path):
                print(f"Directory: {root}")
                for f in files:
                    print(f" - {f}")
                
            return None
                
        except Exception as e:
            print(f"Error in get_filing_path: {str(e)}")
            return None
        
    def check_existing_filing(self, ticker, filing_type):
        """Check if filing already exists and is accessible"""
        base_path = os.path.join("sec-edgar-filings", ticker, filing_type)
        if os.path.exists(base_path):
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if subdirs:
                cik_dir = subdirs[-1]  # Get the most recent filing
                full_path = os.path.join(base_path, cik_dir)
                if (os.path.exists(os.path.join(full_path, "primary-document.html")) or 
                    os.path.exists(os.path.join(full_path, "full-submission.txt"))):
                    print(f"Found existing filing for {ticker} in {full_path}")
                    return True
        return False

    def download_filing(self, ticker, filing_type="10-K"):
        """Download SEC filing for a given ticker"""
        try:
            print(f"Starting download for {ticker} {filing_type}")
        
            # Check if filing already exists
            if self.check_existing_filing(ticker, filing_type):
                print(f"Using existing filing for {ticker}")
                return True
            
            print(f"Starting download for {ticker} {filing_type}")
            
            # Calculate date range for the past year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            print(f"Downloading {filing_type} for {ticker}...")
            downloaded = self.downloader.get(filing_type, ticker, 
                                          after=start_date.strftime("%Y-%m-%d"),
                                          before=end_date.strftime("%Y-%m-%d"),
                                          download_details=True)
            
            print(f"Download completed. Verifying files...")
            
            # Get and verify the filing path
            filing_path = self.get_filing_path(ticker, filing_type)
            if filing_path:
                print(f"Successfully found filing at: {filing_path}")
                return True
            else:
                print("Filing not found after download")
                return False
                
        except Exception as e:
            print(f"Error downloading {filing_type} for {ticker}: {str(e)}")
            return False

    def extract_text(self, file_path):
        """Extract text from SEC filing"""
        try:
            print(f"Starting text extraction from: {file_path}")
            
            # Read the file with error handling for different encodings
            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        print(f"Successfully read file with {encoding} encoding")
                        break
                except UnicodeDecodeError:
                    print(f"Failed to read with {encoding} encoding, trying next...")
                    continue
            else:
                print("Failed to read file with any encoding")
                return None

            # Process with BeautifulSoup
            print("Processing with BeautifulSoup...")
            soup = bs4.BeautifulSoup(content, 'html.parser')
            
            # Remove unnecessary elements
            for element in soup(["script", "style", "table"]):
                element.decompose()
                
            text = soup.get_text(separator=' ')
            text = ' '.join(text.split())
            
            print(f"Extracted {len(text)} characters of text")
            return text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None

    def analyze_sentiment(self, text, chunk_size=512):
        """Analyze sentiment of text using FinBERT"""
        try:
            print(f"Analyzing sentiment of {len(text)} characters of text")
            # Split text into chunks that fit the model's max length
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            sentiments = []
            
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:  # Print progress every 10 chunks
                    print(f"Processing chunk {i+1}/{len(chunks)}")
                    
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiments.append(probabilities[0].detach().numpy())
            
            # Average the sentiments across all chunks
            avg_sentiment = {
                "positive": float(sum(s[2] for s in sentiments)) / len(sentiments),
                "negative": float(sum(s[0] for s in sentiments)) / len(sentiments),
                "neutral": float(sum(s[1] for s in sentiments)) / len(sentiments)
            }
            
            print("Sentiment analysis complete")
            return avg_sentiment
            
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return None

    def analyze_filing(self, ticker, filing_type="10-K"):
        """Main method to analyze a filing"""
        print(f"\nStarting analysis for {ticker} {filing_type}")
        
        # Download the filing
        if not self.download_filing(ticker, filing_type):
            return None

        # Get the filing path
        file_path = self.get_filing_path(ticker, filing_type)
        if not file_path:
            print("Could not find downloaded filing")
            return None
        
        # Extract text
        text = self.extract_text(file_path)
        if text is None:
            return None

        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        if sentiment is None:
            return None

        result = {
            "ticker": ticker,
            "filing_type": filing_type,
            "sentiment": sentiment,
            "file_path": file_path
        }
        
        print(f"\nAnalysis complete for {ticker} {filing_type}")
        return result


def list_downloaded_files(ticker, filing_type):
    path = f"./sec-edgar-filings/{ticker}/{filing_type}"
    if os.path.exists(path):
        print(f"Files in {path}:")
        for root, dirs, files in os.walk(path):
            for file in files:
                print(f" - {file}")
    else:
        print(f"No directory found at {path}")

# Usage example
if __name__ == "__main__":
    # Replace these with your information
    MY_COMPANY = "Sanch"
    MY_EMAIL = "sanchit_sharma7@hotmail.com"
    
    analyzer = SECAnalyzer(MY_COMPANY, MY_EMAIL)
    ticker = "RBLX"  # or "RBLX" or any other ticker
    print("\n" + "="*50)
    print(f"Analyzing {ticker} 10-K filing")
    print("="*50)

    # Debug: Print full directory structure
    base_path = os.path.join("sec-edgar-filings", ticker, "10-K")
    print("\nFull directory structure:")
    for root, dirs, files in os.walk(base_path):
        print(f"\nDirectory: {root}")
        if dirs:
            print("Subdirectories:", dirs)
        if files:
            print("Files:", files)
    
    result = analyzer.analyze_filing(ticker, "10-K")
    
    if result:
        print("\nFinal Result:")
        print(f"Ticker: {result['ticker']}")
        print(f"Filing Type: {result['filing_type']}")
        print("Sentiment Scores:")
        for key, value in result['sentiment'].items():
            print(f"  {key}: {value:.4f}")
        print(f"File analyzed: {result['file_path']}")
    else:
        print("\nAnalysis failed.")