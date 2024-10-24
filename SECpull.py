from sec_edgar_downloader import Downloader
import bs4
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lime.lime_text import LimeTextExplainer
import gc


import os



class SECAnalyzer:
    def __init__(self, company_name, email_address):
        self.downloader = Downloader(company_name, email_address)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.explainer = LimeTextExplainer(class_names=['negative', 'netural', 'positive'])
        # Set device for GPU usage if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

              # Configure batch sizes and limits
        self.batch_size = 8
        self.max_explanations = 10  # Limit number of explanations to prevent memory issues
        
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

            #We also need to remove XBRL tags as I noticed it won't print the text content properly otherwise (non-narrative elements)
            for element in soup(["script", "style"]):
                element.decompose()
            
            # find all the text elements 
            text_elements = []

            #finding the common SEC filing text containers, because I want to model to focus on these parts:
            # List of keywords that indicate important sections
            important_sections = [
                "Business", "Management's Discussion", "Risk Factors",
                "Operating Results", "Financial Condition", "Critical Accounting",
                "Market Risk", "Controls and Procedures", "Executive Officers",
                "Competition", "Properties", "Revenue", "Growth", "Strategy",
                "Operations", "Customers", "Technology"
            ]
            # Find sections with these keywords
            for section in soup.find_all(['div', 'p', 'td', 'span']):
                text = section.get_text().strip()
                
                # Skip short texts and obvious headers/metadata
                if len(text) < 100:
                    continue
                    
                # Skip if contains XBRL or form metadata
                if any(marker in text.lower() for marker in [
                    'gaap:', 'xbrl:', 'cik', 'xmlns:', '☐', '☒',
                    'check mark', 'indicate by', 'form', 'pursuant to'
                ]):
                    continue
                
                # Check if this section contains important content
                if any(keyword.lower() in text.lower() for keyword in important_sections):
                    # Clean up the text
                    cleaned_text = ' '.join(text.split())
                    if len(cleaned_text.split()) > 20:  # Only keep substantial paragraphs
                        text_elements.append(cleaned_text)
            
            full_text = ' '.join(text_elements)
            
            print(f"Extracted {len(full_text)} characters of meaningful text")
            if full_text:
                print("\nSample of extracted text:")
                print(full_text[:500] + "...\n")
            else:
                print("No meaningful text extracted")
                
            return full_text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

            
    def explain_sentiment(self, text_chunk):
        """Add explainability to show why the model made certain decisions"""
        explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
        exp = explainer.explain_instance(text_chunk, self.predict_proba)
        
        # Highlight influential phrases
        key_phrases = exp.as_list()
        return {
            'text': text_chunk,
            'key_phrases': key_phrases,
            'impact_weights': {phrase: weight for phrase, weight in key_phrases}
        }
    
    def lime_explain_chunk(self, text_chunk, num_features = 10):
        """Explains the predictions for a specific chunk of text"""
        """
           Explain predictions for a specific chunk of text
        
           Parameters:
            text_chunk: The text to explain
            num_features: Number of important features (words) to return
            
           Returns:
            Dictionary containing the text and its important words with their scores
        """
        try:
            exp = self.explainer.explain_instance(
            text_chunk, self.predict_proba, num_features=num_features, num_samples=500) # num of samples will be reduced for better performance
        
            word_importance = exp.as_list() #creates tuples containing word importance

        #create a simple dictionary

            explanation = {
                'important_words': [
                    {
                        'word': word,
                        'importance': score,
                        'sentiment': 'positive' if score > 0 else 'negative',
                        'magnitude': abs(score)
                    }
                    for word, score in word_importance
                ]
            }


            # Clear some memory
            del exp
            gc.collect()
            
            return explanation
            
        except Exception as e:
            print(f"Error in lime_explain_chunk: {str(e)}")
            return None
    


    def predict_proba(self, texts):
        """
        Memory-safe prediction function for LIME that processes texts in small batches
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Process in small batches of 8 texts at a time
        batch_size = 8
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
            
            # Clear memory after each batch
            del inputs, outputs, probs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        return np.array(all_probs)

    def analyze_sentiment(self, text, chunk_size=512):
        """Analyze sentiment of text using FinBERT"""
        try:
            print(f"Analyzing sentiment of {len(text)} characters of text")
            
            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            sentiments = []
            explanations = []
            
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:
                    print(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    # Get sentiment for single chunk
                    chunk_sentiment = self.predict_proba([chunk])[0]
                    sentiments.append(chunk_sentiment)
                    
                    # Only get LIME explanation for high confidence chunks
                    max_prob = max(chunk_sentiment)
                    if max_prob > 0.7 and len(explanations) < 10:  # Limit explanations to 10
                        # Use LIME with reduced complexity
                        exp = self.explainer.explain_instance(
                            chunk,
                            self.predict_proba,
                            num_features=5,  # Reduced features
                            num_samples=50   # Significantly reduced samples
                        )
                        
                        # Extract important words
                        word_importance = exp.as_list()
                        explanations.append({
                            'text': chunk[:100] + "...",  # Just store the start
                            'important_words': [
                                {'word': word, 'score': score}
                                for word, score in word_importance
                            ]
                        })
                        
                        # Clear LIME explanation from memory
                        del exp
                        
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    continue
                
                # Clear memory after each chunk
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate average sentiment
            if sentiments:
                avg_sentiment = {
                    "positive": float(sum(s[2] for s in sentiments)) / len(sentiments),
                    "negative": float(sum(s[0] for s in sentiments)) / len(sentiments),
                    "neutral": float(sum(s[1] for s in sentiments)) / len(sentiments)
                }
                
                return {
                    "sentiment_scores": avg_sentiment,
                    "explanations": explanations
                }
            else:
                return None
            
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
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
    analyzer.batch_size = 8  # Adjust based on your available memory
    analyzer.max_explanations = 10  # Limit number of explanations
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
        for key, value in result['sentiment']['sentiment_scores'].items():
            print(f"  {key}: {value:.4f}")
        print("\nKey Explanations:")
        for exp in result['sentiment']['explanations']:
            print("\nChunk:", exp['text'])
            print("Important words:")
            for word_info in exp['important_words']:
                print(f"  {word_info['word']}: {word_info['score']:.4f}")


        print(f"File analyzed: {result['file_path']}")
    else:
        print("\nAnalysis failed.")