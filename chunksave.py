import json
from datetime import datetime

class ChunkAnalyzer:
    def save_chunks(result, output_file=None):
        """Save all chunks and their analysis to a file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"chunk_analysis_{result['ticker']}_{timestamp}.json"
        
        analysis_data = {
            "ticker": result['ticker'],
            "filing_type": result['filing_type'],
            "analysis_date": datetime.now().isoformat(),
            "chunks": []
        }
        
        for i, chunk in enumerate(result['sentiment']['all_chunks']):
            chunk_data = {
                "chunk_id": i,
                "text": chunk,
                "sentiment": result['sentiment']['all_sentiments'][i].tolist() 
                    if i < len(result['sentiment']['all_sentiments']) else None,
                "has_detailed_analysis": any(
                    exp['text'] == chunk 
                    for exp in result['sentiment']['explanations']
                )
            }
            
            # Add LIME analysis if available
            matching_explanations = [
                exp for exp in result['sentiment']['explanations'] 
                if exp['text'] == chunk
            ]
            if matching_explanations:
                chunk_data["important_words"] = matching_explanations[0]['important_words']
            
            analysis_data["chunks"].append(chunk_data)
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return output_file
