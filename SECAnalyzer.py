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
        """Extract text from key narrative sections while excluding boilerplate"""
        try:
            print(f"Starting text extraction from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            print("Processing with BeautifulSoup...")
            soup = bs4.BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            text_elements = []
            
            # Define sections to specifically exclude
            exclude_sections = [
                "forward-looking statements",
                "safe harbor",
                "market value",
                "common stock",
                "pursuant to",
                "form 10-k",
                "table of contents",
                "exhibits",
                "signatures"
            ]
            
            # Define high-value sections and their identifying phrases
            target_sections = {
                "Financial Performance": [
                    "net revenue increased",
                    "net revenue decreased",
                    "financial results",
                    "operating results",
                    "financial performance",
                    "fiscal year"
                ],
                "Business Strategy": [
                    "our strategy",
                    "business strategy",
                    "competitive advantage",
                    "growth strategy",
                    "market opportunity"
                ],
                "Risk Analysis": [
                    "primary risks",
                    "key challenges",
                    "competitive pressures",
                    "market conditions",
                    "regulatory environment"
                ],
                "Operations": [
                    "daily active users",
                    "platform engagement",
                    "user metrics",
                    "engagement metrics",
                    "developer community"
                ]
            }
            
            # Process each paragraph
            for section in soup.find_all(['div', 'p']):
                text = section.get_text().strip()
                
                # Skip if too short or contains excluded content
                if len(text) < 200 or any(exclude in text.lower() for exclude in exclude_sections):
                    continue
                
                # Check if text contains any target phrases
                for section_type, phrases in target_sections.items():
                    if any(phrase.lower() in text.lower() for phrase in phrases):
                        # Clean the text
                        cleaned_text = ' '.join(text.split())
                        if len(cleaned_text.split()) > 30:  # Ensure substantial content
                            print(f"Found {section_type} section")
                            text_elements.append(cleaned_text)
                        break
            
            # Additional check for MD&A section
            mda_markers = soup.find_all(string=lambda text: text and "Management's Discussion and Analysis" in text)
            if mda_markers:
                for marker in mda_markers:
                    current = marker.find_parent().find_next_sibling()
                    while current and len(text_elements) < 20:  # Limit to prevent over-collection
                        if current.name in ['p', 'div']:
                            text = current.get_text().strip()
                            if len(text) > 200 and not any(exclude in text.lower() for exclude in exclude_sections):
                                cleaned_text = ' '.join(text.split())
                                if len(cleaned_text.split()) > 30:
                                    print("Found MD&A section")
                                    text_elements.append(cleaned_text)
                        current = current.find_next_sibling()
            
            full_text = ' '.join(text_elements)
            
            print(f"Extracted {len(full_text)} characters of meaningful text")
            print(f"Found {len(text_elements)} relevant sections")
            
            if full_text:
                print("\nSample of extracted text (first section):")
                print(text_elements[0][:500] + "...\n")
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

    def analyze_sentiment(self, text, chunk_size=2048):  # Increased from 512
        """Analyze sentiment with better context preservation"""
        try:
            print(f"Analyzing sentiment of {len(text)} characters of text")
            
            # Split into sentences first
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Combine sentences into chunks while preserving context
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                # Add sentence length plus space
                sentence_length = len(sentence) + 1
                
                if current_length + sentence_length > chunk_size:
                    # If this chunk is getting full, save it
                    if current_chunk:
                        chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            
            sentiments = []
            explanations = []
            
            print(f"Processing {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                if i % 5 == 0:  # Print progress less frequently
                    print(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Skip chunks that are mostly numbers or technical markers
                if sum(c.isdigit() for c in chunk) / len(chunk) > 0.2:
                    continue
                    
                try:
                    # Get sentiment for chunk
                    inputs = self.tokenizer(
                        chunk,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        sentiment = probabilities[0].cpu().numpy()
                        sentiments.append(sentiment)
                    
                    # Only get LIME explanation for significant chunks
                    if max(sentiment) > 0.6 and len(explanations) < 10:
                        exp = self.explainer.explain_instance(
                            chunk,
                            self.predict_proba,
                            num_features=8,  # Increased from 5
                            num_samples=50
                        )
                        
                        word_importance = exp.as_list()
                        explanations.append({
                            'text': chunk,  # Store full chunk for context
                            'important_words': [
                                {'word': word, 'score': score}
                                for word, score in word_importance
                            ]
                        })
                        
                        del exp
                        
                    # Clear memory
                    del inputs, outputs, probabilities
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            
            # Calculate average sentiment
            if sentiments:
                avg_sentiment = {
                    "positive": float(sum(s[2] for s in sentiments)) / len(sentiments),
                    "negative": float(sum(s[0] for s in sentiments)) / len(sentiments),
                    "neutral": float(sum(s[1] for s in sentiments)) / len(sentiments)
                }
                
                return {
                    "sentiment_scores": avg_sentiment,
                    "explanations": explanations,
                    "all_chunks": chunks,           # Added for human annotation
                    "all_sentiments": sentiments    # Added for human annotation
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


pass