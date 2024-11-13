    # Initialize components
from SECAnalyzer import SECAnalyzer
from finvalidation import SentimentBenchmark
from hannotations import AnnotationHandler
from pathlib import Path
from datetime import datetime
import json

def main():
    # Initialize components
    analyzer = SECAnalyzer("YourCompany", "your@email.com")
    annotation_handler = AnnotationHandler()
    benchmark = SentimentBenchmark()
    # Analyze filing
    ticker = "RBLX"
    result = analyzer.analyze_filing(ticker, "10-K")
    
    # Create human validation tasks
    if result:
        # Load annotations
        annotation_path = Path("ticker_annotations") / f"annotations_{ticker}_20241112_025904.json"
        annotations = annotation_handler.load_annotations(annotation_path)
        
        # Get chunk-level predictions
        chunk_predictions = [
            max(exp['important_words'], key=lambda x: abs(x['score']))['score']
            for exp in result['sentiment']['explanations']
        ]
        
        # Ensure we're comparing the same number of items
        min_length = min(len(chunk_predictions), len(annotations))
        chunk_predictions = chunk_predictions[:min_length]
        annotations = annotations[:min_length]
        
        # Get comprehensive validation metrics
        validation_metrics = benchmark.calculate_ml_metrics(
            predictions=chunk_predictions,
            actual_values=[ann['sentiment'] for ann in annotations]
        )
        
        # Get statistical validation
        statistical_metrics = benchmark.calculate_statistical_metrics(
            predictions=chunk_predictions,
            actual_values=[ann['sentiment'] for ann in annotations]
        )
        
        # Enhanced sentiment analysis dictionary
        sentiment_analysis = {
            'chunk_level': {
                'predictions': chunk_predictions,
                'validation': benchmark.compare_with_human(
                    model_sentiment={
                        'explanations': result['sentiment']['explanations'][:min_length]
                    },
                    human_annotations=annotations
                )
            },
            'overall': {
                'sentiment_scores': result['sentiment']['sentiment_scores'],
                'timestamp': datetime.now().isoformat()
            },
            'statistical_validation': statistical_metrics,
            'ml_metrics': validation_metrics
        }
        
        # Print comprehensive results
        print("\nValidation Results:")
        print(f"Analyzed {min_length} chunks")
        
        print("\nStatistical Metrics:")
        print(f"Mean Difference: {statistical_metrics['mean_difference']:.3f}")
        print(f"Standard Deviation: {statistical_metrics['std_deviation']:.3f}")
        print(f"Confidence Interval: {statistical_metrics['confidence_interval']}")
        print(f"Statistically Significant: {statistical_metrics['statistically_significant']}")
        
        print("\nML Metrics:")
        print(f"Mean Absolute Error: {validation_metrics['mae']:.3f}")
        print(f"Root Mean Square Error: {validation_metrics['rmse']:.3f}")
        print(f"Correlation: {validation_metrics['correlation']:.3f}")
        print(f"Agreement Rate: {validation_metrics['agreement_rate']:.3f}")
        print(f"Directional Accuracy: {validation_metrics['directional_accuracy']:.3f}")

        print("\nDifference Distribution:")
        print(f"Mean Difference: {validation_metrics['difference_distribution']['mean']:.3f}")
        print(f"Std of Differences: {validation_metrics['difference_distribution']['std']:.3f}")
        print(f"Max Difference: {validation_metrics['difference_distribution']['max_diff']:.3f}")
        print(f"Min Difference: {validation_metrics['difference_distribution']['min_diff']:.3f}")
        
        print("\nConfusion Matrix:")
        print(validation_metrics['confusion_matrix'])

        

if __name__ == "__main__":
    main()