import json
from datetime import datetime
import random
from pathlib import Path


class AnnotationHandler:

    def __init__(self, storage_dir="ticker_annotations"):
        #This will create a new ticket_annotations directory if it does not exist
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)


    def sample_chunks_for_annotations(self, chunks, n_samples=10):
        "Select diverse chunks for annotations"
        important_chunks = [chunk for chunk in chunks if any(keyword in chunk.lower() for keyword in ["revenue", "earnings", "growth", "inventory", "expenses", "investments", "patents", "trademark", "property", "cash", "accounts", "dividends", "debt", "shareholder", "equity", "asset", "depreciation", "invest", "repurchase", "paid"])]

        # Debug print
        print(f"Found {len(important_chunks)} important chunks")

        #Now get remaining chunks 

        remaining_chunks = [c for c in chunks if c not in important_chunks]
        print(f"Remaining chunks: {len(remaining_chunks)}")
        # Calculate how many random chunks we need, max function ensures we dont go negative
        needed_random = max(0, n_samples - len(important_chunks))
        print(f"Need {needed_random} additional random chunks")

        #If we need more random chunks than available, take all the remaining chunks
        if needed_random > len(remaining_chunks):
            random_chunks = remaining_chunks
        else:
            # Otherwise, sample the requested amount
            random_chunks = random.sample(
            remaining_chunks,
            needed_random)
        selected_chunks = important_chunks + random_chunks

        print(f"\nSelected {len(selected_chunks)} chunks for annotation:")
        print(f"- {len(important_chunks)} important chunks")
        print(f"- {len(random_chunks)} random chunks")
        
        return selected_chunks


    def annotate_chunks(self, chunks):
        annotations = []
        print("\nAnnotation Guidelines:")
        print("-1: Very Negative")
        print("0: Neutral")
        print("1: Very Positive")
        print("Ctrl+C to stop annotating\n")
        

        for i, chunk in enumerate(chunks):
            try:
                print(f"\nChunk {i+1}/{len(chunks)}:")
                print("="*50)
                print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                print("="*50 + "\n")

                sentiment = float(input("Sentiment (-1 to 1): "))
                while not -1 <= sentiment <= 1:
                    print("Please enter a value between -1 and 1")
                    sentiment = float(input("Sentiment (-1 to 1): "))
                    
                confidence = int(input("Confidence (1-5): "))
                while not 1 <= confidence <= 5:
                    print("Please enter a value between 1 and 5")
                    confidence = int(input("Confidence (1-5): "))
                    
                notes = input("Notes (optional): ")
                
                annotations.append({
                    'chunk_id': i,
                    'text': chunk,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'notes': notes,
                    'annotation_date': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\nAnnotation process stopped by user")
                break
        
        return annotations
    
    def save_annotations(self, ticker, annotations):
        "Save annotations to a JSON file"
    
        filename = self.storage_dir / f"annotations_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"\nAnnotations saved to: {filename}")
        return filename
    

    def load_annotations(self, filename):
        "Load annotations from a file"
        with open(filename, 'r') as f:
            return json.load(f)
        

    def create_annotations(self, chunks, ticker, n_samples=10):
        "Complete this annotation workflow"

        #Sample chunks

        selected_chunks = self.sample_chunks_for_annotations(chunks, n_samples)

        #Get Annotations
        annotations = self.annotate_chunks(selected_chunks)

        #Save Annotations
        if annotations:
            filename = self.save_annotations(ticker, annotations)
            return filename, annotations
        return None, None
    
