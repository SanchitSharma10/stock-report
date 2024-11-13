from sklearn.model_selection import KFold
from typing import Dict, List, Any
import numpy as np
import pandas as pd

class CrossValidator:
    def __init__(self, n_splits=5):
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.validation_metrics = {
            'sentiment_accuracy': [],
            'financial_correlation': [],
            'human_agreement': [],
            'consistency_score': []
        }

    def validate_model(self, text_chunks: List[str], human_annotations: List[Dict], financial_metrics: Dict[str, float], model_predictions: List[float]) -> Dict[str,Any]:
        # The SEC filing text broken into chunks will be text_chunks
        # The human annotations for each chunk will be human ratings of these chunks
        # The financial metrics for each chunk will be the actual financial data
        # The model predictions will be what FinBert ProsusAI predicted

        try: 
            #NP arrays are not only better supported for sci-kit library, but also are implemented in C and optimized for numerical operations
            chunk_indices = np.array(range(len(text_chunks)))

            for train_idx, test_idx in self.kf.split(chunk_indices): # basically obtain the values for test and train in the KF splitting of the text_chunks
                #Get test data for this fold
                test_chunks = [text_chunks[i] for i in test_idx]
                test_human_annotations = [human_annotations[i] for i in test_idx]
                test_model_predictions = [model_predictions[i] for i in test_idx]

                # Calculate the metrics now for this same fold

                fold_metrics = {
                    'sentiment_accuracy': self.calculate_sentiment_accuracy(test_human_annotations, test_model_predictions),
                    'financial_correlation': self.calculate_financial_correlation(test_chunks, financial_metrics, test_model_predictions),
                    'human_agreement': self.calculate_human_agreement(test_human_annotations, test_model_predictions),
                    'consistency_score': self.calculate_consistency_score(test_chunks, test_model_predictions)
                }

                #Append the metrics for this fold
                for metric, value in fold_metrics.items():
                    self.validation_metrics[metric].append(value)

            return self._calculate_final_results()

        
        except Exception as e:
            print(f"Error in validating_model_performance: {str(e)}")
            return None

    def _calculate_sentiment_accuracy(self, predictions: List[float], annotations: List[Dict])-> float:
        return np.mean([abs(pred - ann['sentiment']) < 0.2 for pred, ann in zip(predictions, annotations)
        ])
    
    def _calculate_human_agreement(self, predictions: List[float], annotations: List[Dict])-> float:
        """ What we need to do is caluclate the predictions and human annotations together"""

        agreements = []

        for pred, ann in zip(predictions, annotations):
            #Whats the base agreement; how close is this prediction to human rating
            agreement = 1 - abs(pred - ann['sentiment'])

            #we weigh the human confidnece higher = more important
            confidence_weight = ann['confidence'] / 5.0 # This will normalize the value to 0-1

            #Weighted agreement
            weighted_agreement = agreement * confidence_weight
            agreements.append(weighted_agreement)


        #Return average weighted agreement
        return np.mean(agreements) if agreements else 0.0

