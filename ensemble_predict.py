from cnn_predict import predict_cnn
from bert_predict import predict_bert

def predict_ensemble(text, models=None, weights=None):
    """
    Ensemble prediction combining results from CNN and BERT models.
    
    Args:
        text (str): The text to analyze
        models (list, optional): List of models to use ['CNN', 'BERT']. Defaults to both.
        weights (dict, optional): Weights for each model. Defaults to equal weights.
        
    Returns:
        tuple: (prediction, details) where details contains each model's prediction and confidence
    """
    if models is None:
        models = ['CNN', 'BERT']
    
    if weights is None:
        # Default equal weights
        weights = {model: 1/len(models) for model in models}
    
    results = {}
    all_scores = {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    # Get predictions from models
    for model in models:
        if model == 'CNN':
            pred, scores = predict_cnn(text, return_scores=True)
            results['CNN'] = {'prediction': pred, 'scores': scores}
            
            # Add weighted scores to ensemble
            for sentiment, score in scores.items():
                all_scores[sentiment] += score * weights['CNN']
                
        elif model == 'BERT':
            pred, scores = predict_bert(text, return_scores=True)
            results['BERT'] = {'prediction': pred, 'scores': scores}
            
            # Add weighted scores to ensemble
            for sentiment, score in scores.items():
                all_scores[sentiment] += score * weights['BERT']
    
    # Determine final prediction
    final_prediction = max(all_scores, key=all_scores.get)
    
    # Include detailed results for transparency
    details = {
        'individual_predictions': results,
        'ensemble_scores': all_scores,
        'final_prediction': final_prediction
    }
    
    return final_prediction, details

if __name__ == "__main__":
    sample = input("Enter a review: ")
    prediction, details = predict_ensemble(sample)
    
    print(f"Ensemble Prediction: {prediction}")
    print("\nIndividual model predictions:")
    
    for model, result in details['individual_predictions'].items():
        print(f"- {model}: {result['prediction']}")
    
    print("\nEnsemble scores:")
    for sentiment, score in details['ensemble_scores'].items():
        print(f"- {sentiment}: {score:.4f}")