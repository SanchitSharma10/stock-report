from scipy import stats
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from typing import List, Dict, Tuple, Any

class SentimentBenchmark:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'financial_alignment', 'human_agreement',
            'statistical_significance', 'reliability'
        ]

    def calculate_statistical_metrics(
            self,
            predictions: List[float],
            actual_values: List[float],
    ) -> Dict[str, float]:
        "Statistical validation metrics"
        differences = np.array(predictions) - np.array(actual_values)

        confidence_interval = stats.t.interval(
            alpha = self.confidence_level,
            df= len(differences)-1,
            loc= np.mean(differences),
            scale=stats.sem(differences)

        )

        t_stat, p_value =  stats.ttest_1samp(differences, 0)

        return {
            'mean_difference': float(np.mean(differences)),
            'std_deviation': float(np.std(differences)),
            'confidence_interval': tuple(map(float, confidence_interval)),
            'p_value': float(p_value),
            'statistically_significant': p_value < (1 - self.confidence_level)
        }
    
    def compare_with_human(
        self,
        model_sentiment: Dict,
        human_annotations: List[Dict]
    ) -> Dict[str, Any]:
        """Enhanced human comparison with statistical validation"""
        try:
            # Extract scores (keeping your original logic)
            model_scores = []
            human_scores = []
            
            for model_exp, human_ann in zip(
                model_sentiment['explanations'],
                human_annotations
            ):
                chunk_sentiment = max([
                    word_info['score'] 
                    for word_info in model_exp['important_words']
                ])
                
                model_scores.append(chunk_sentiment)
                human_scores.append(human_ann['sentiment'])
            
            # Calculate comprehensive metrics
            results = {
                # Original metrics
                'correlation': stats.pearsonr(model_scores, human_scores)[0],
                'mean_difference': np.mean(np.abs(
                    np.array(model_scores) - np.array(human_scores)
                )),
                'agreement_rate': np.mean(
                    np.abs(np.array(model_scores) - np.array(human_scores)) < 0.2
                ),
                'confidence_weighted_agreement': self._calculate_confidence_weighted_agreement(
                    model_scores,
                    human_scores,
                    [h['confidence'] for h in human_annotations]
                ),
                
                # New statistical metrics
                'statistical_validation': self.calculate_statistical_metrics(
                    model_scores, human_scores
                ),
                
                # ML metrics
                'ml_metrics': self.calculate_ml_metrics(
                    model_scores, human_scores
                )
            }
            
            # Add interpretation
            results['interpretation'] = self._interpret_results(results)
            
            return results
            
        except Exception as e:
            print(f"Error in enhanced comparison: {str(e)}")
            return None
        

    
    def benchmark_against_financials(
        self,
        sentiment_results: Dict,
        financial_metrics: Dict
    ) -> Dict[str, Any]:
         try:
            """Compare sentiment with actual financial performance"""
            correlation = stats.pearsonr(
                sentiment_results['sentiment_scores'],
                financial_metrics['performance']
            )[0]
            
            results = {
                'correlation': correlation,
                'alignment': correlation > 0.7,
                'statistical_significance': self.calculate_statistical_metrics(
                    sentiment_results['sentiment_scores'],
                    financial_metrics['performance']
                )
            }
         
            return results
            
         except Exception as e:
            print(f"Error in financial benchmarking: {str(e)}")
            return None
         
    def calculate_ml_metrics(
        self,
        predictions: List[float],
        actual_values: List[float],
        threshold: float = 0.2 #Threshold for considering differences signifcant
    ) -> Dict[str, Any]:
        """
        Calculate machine learning specific metrics
        """
        # Continous variables
        preds = np.array(predictions)
        actuals = np.array(actual_values)
        
        # Calculate mae, rmse, correlation
        metrics = {
        'mae': np.mean(np.abs(preds - actuals)),  # Mean Absolute Error
        'rmse': np.sqrt(np.mean((preds - actuals)**2)),  # Root Mean Square Error
        'correlation': np.corrcoef(preds, actuals)[0,1],  # Correlation coefficient
        
        # Calculate agreement rate (within threshold)
        'agreement_rate': np.mean(np.abs(preds - actuals) < threshold),
        
        # Directional accuracy (do they agree on positive/negative)
        'directional_accuracy': np.mean(np.sign(preds) == np.sign(actuals)),
        
        # Distribution of differences
        'difference_distribution': {
            'mean': float(np.mean(preds - actuals)),
            'std': float(np.std(preds - actuals)),
            'max_diff': float(np.max(np.abs(preds - actuals))),
            'min_diff': float(np.min(np.abs(preds - actuals)))
            }
        }
    
        return metrics
    
    def _calculate_confidence_weighted_agreement(
        self,
        model_scores: List[float],
        human_scores: List[float],
        confidence_scores: List[int]
    ) -> float:
        """Your existing confidence weighted agreement calculation"""
        return float(np.average(
            np.abs(np.array(model_scores) - np.array(human_scores)) < 0.2,
            weights=np.array(confidence_scores) / 5.0
        ))
    
    def _interpret_results(self, results: Dict) -> str:
        """Enhanced interpretation"""
        interpretation = []
        
        # Original interpretations
        if results['correlation'] > 0.7:
            interpretation.append("Strong agreement with human annotations")
        elif results['correlation'] > 0.5:
            interpretation.append("Moderate agreement with human annotations")
        else:
            interpretation.append("Weak agreement with human annotations")
        
        # Add statistical significance interpretation
        if results['statistical_validation']['statistically_significant']:
            interpretation.append("Results are statistically significant")
        
        return " | ".join(interpretation)
    
    def calculate_reliability(
        self,
        model_predictions: List[float],
        human_annotations: List[Dict],
        confidence_scores: List[int] = None
    ) -> Dict[str, float]:
        """
        Calculate reliability metrics including:
        - Inter-rater reliability
        - Model consistency
        - Confidence-weighted reliability
        """
        reliability_metrics = {
            'inter_rater': self._calculate_confidence_weighted_agreement(
                model_predictions,
                [ann['sentiment'] for ann in human_annotations],
                confidence_scores if confidence_scores else [ann['confidence'] for ann in human_annotations]
            ),
            'model_consistency': np.std(model_predictions),  # Lower std = more consistent
            'reliability_level': 'Not calculated'  # Will be set below
        }
        
        # Determine reliability level
        avg_reliability = (reliability_metrics['inter_rater'] + 
                        (1 - reliability_metrics['model_consistency']))  # Convert std to reliability
        
        if avg_reliability > 0.8:
            reliability_metrics['reliability_level'] = 'High'
        elif avg_reliability > 0.6:
            reliability_metrics['reliability_level'] = 'Medium'
        else:
            reliability_metrics['reliability_level'] = 'Low'
            
        return reliability_metrics
    
    def compare_models(self, text, models):
        """Compare results across different models"""
        results = {}
        for name, model in models.items():
            results[name] = model.analyze_sentiment(text)
        
        return self._calculate_model_agreement(results)
    
    def compare_with_human(self, model_sentiment, human_annotations):
        """
        Compare model sentiment results with human annotations
        
        Parameters:
        model_sentiment: dict with 'sentiment_scores' and 'explanations'
        human_annotations: list of dicts with 'sentiment' and 'confidence'
        """
        try:
            # Extract model scores and human scores
            model_scores = []
            human_scores = []
            
            # Match chunks with human annotations
            for i, (model_exp, human_ann) in enumerate(
                zip(model_sentiment['explanations'], human_annotations)
            ):
                # Get model sentiment for this chunk
                chunk_sentiment = max([
                    word_info['score'] 
                    for word_info in model_exp['important_words']
                ])
                
                model_scores.append(chunk_sentiment)
                human_scores.append(human_ann['sentiment'])
            
            # Calculate agreement metrics
            results = {
                # Correlation between model and human scores
                'correlation': stats.pearsonr(model_scores, human_scores)[0],
                
                # Average absolute difference
                'mean_difference': np.mean(np.abs(
                    np.array(model_scores) - np.array(human_scores)
                )),
                
                # Agreement rate (within 0.2 difference)
                'agreement_rate': np.mean(
                    np.abs(np.array(model_scores) - np.array(human_scores)) < 0.2
                ),
                
                # Consider human confidence
                'confidence_weighted_agreement': self._calculate_confidence_weighted_agreement(
                    model_scores,
                    human_scores,
                    [h['confidence'] for h in human_annotations]
                )
            }
            
            # Add interpretation
            results['interpretation'] = self._interpret_results(results)
            
            return results
            
        except Exception as e:
            print(f"Error in compare_with_human: {str(e)}")
            return None
    
    def _calculate_confidence_weighted_agreement(
        self, 
        model_scores, 
        human_scores, 
        confidence_scores
    ):
        """Calculate agreement weighted by human confidence"""
        differences = np.abs(np.array(model_scores) - np.array(human_scores))
        weights = np.array(confidence_scores) / 5.0  # Normalize to 0-1
        
        return float(np.average(
            differences < 0.2,  # Agreement threshold
            weights=weights
        ))
    
    def _interpret_results(self, results):
        """Provide interpretation of the comparison results"""
        interpretation = []
        
        # Interpret correlation
        if results['correlation'] > 0.7:
            interpretation.append("Strong agreement with human annotations")
        elif results['correlation'] > 0.5:
            interpretation.append("Moderate agreement with human annotations")
        else:
            interpretation.append("Weak agreement with human annotations")
        
        # Interpret agreement rate
        if results['agreement_rate'] > 0.8:
            interpretation.append("High agreement rate")
        elif results['agreement_rate'] > 0.6:
            interpretation.append("Moderate agreement rate")
        else:
            interpretation.append("Low agreement rate")
        
        return " | ".join(interpretation)