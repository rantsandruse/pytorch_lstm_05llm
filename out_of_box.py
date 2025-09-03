from openai import OpenAI
import time
import random
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_imdb_data():
    """Load the same IMDB dataset for comparison"""
    # Use Hugging Face datasets instead of torchtext
    from datasets import load_dataset
    
    dataset = load_dataset('imdb')
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    return train_texts, train_labels, test_texts, test_labels

def batch_classify_sentiment(texts, batch_size=10, model="gpt-4.1-mini"):
    """Classify sentiment using OpenAI API with batch processing"""
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Create numbered list of reviews
        reviews_list = "\n".join([f"{j+1}. {text[:500]}" for j, text in enumerate(batch_texts)])
        
        prompt = f"""You are a sentiment analysis expert. Classify each movie review as positive or negative sentiment.

Reviews:
{reviews_list}

Respond with exactly one number per review (1 for positive, 0 for negative), separated by commas. Example: 1, 0, 1

Classifications:"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(batch_texts) * 5,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().lower()
            predictions = [int(pred.strip()) if pred.strip() in ['0', '1'] else 0 for pred in result.split(",")]
            
            # Handle case where response doesn't match expected format
            if len(predictions) != len(batch_texts):
                print(f"Warning: Expected {len(batch_texts)} predictions, got {len(predictions)}")
                predictions = predictions[:len(batch_texts)] + [0] * (len(batch_texts) - len(predictions))
            
            all_predictions.extend(predictions)
            
        except Exception as e:
            print(f"API Error: {e}")
            # Default to negative for failed batch
            all_predictions.extend([0] * len(batch_texts))
    
    return all_predictions

def batch_few_shot_classify_sentiment(texts, batch_size=10, model="gpt-4.1-mini"):
    """Classify sentiment using few-shot learning with batch processing"""
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Create numbered list of reviews
        reviews_list = "\n".join([f"{j+1}. {text[-500:]}" for j, text in enumerate(batch_texts)])
        
        prompt = f"""Classify movie reviews as positive or negative sentiment.
    
Examples:
Review: "This movie was absolutely fantastic! Great acting and plot."
Sentiment: 1

Review: "Terrible movie, waste of time. Poor acting."
Sentiment: 0

Review: "One of the best films I've ever seen!"
Sentiment: 1

Now classify these reviews:
{reviews_list}

Respond with exactly one number per review (1 for positive, 0 for negative), separated by commas. Example: 1, 0, 1

Classifications:"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(batch_texts) * 5,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().lower()
            predictions = [int(pred.strip()) if pred.strip() in ['0', '1'] else 0 for pred in result.split(",")]
            
            # Handle case where response doesn't match expected format
            if len(predictions) != len(batch_texts):
                print(f"Warning: Expected {len(batch_texts)} predictions, got {len(predictions)}")
                predictions = predictions[:len(batch_texts)] + [0] * (len(batch_texts) - len(predictions))
            
            all_predictions.extend(predictions)
            
        except Exception as e:
            print(f"API Error: {e}")
            # Default to negative for failed batch
            all_predictions.extend([0] * len(batch_texts))
    
    return all_predictions

def evaluate_openai_approach(test_texts, test_labels, sample_size=1000, use_few_shot=False, batch_size=10):
    """Evaluate OpenAI API approach on IMDB dataset with batch processing"""
    # Sample a subset due to API costs and rate limits
    indices = random.sample(range(len(test_texts)), min(sample_size, len(test_texts)))
    sampled_texts = [test_texts[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]
    
    start_time = time.time()
    
    classify_func = batch_few_shot_classify_sentiment if use_few_shot else batch_classify_sentiment
    approach_name = "Few-shot" if use_few_shot else "Zero-shot"
    
    print(f"Starting {approach_name} batch evaluation with {len(sampled_texts)} samples...")
    print(f"Using batch size: {batch_size}")
    
    predictions = classify_func(sampled_texts, batch_size=batch_size)
    
    api_time = time.time() - start_time
    accuracy = accuracy_score(sampled_labels, predictions)
    
    # Find errors for analysis
    errors = []
    for i, (text, true_label, pred_label) in enumerate(zip(sampled_texts, sampled_labels, predictions)):
        if true_label != pred_label:
            errors.append({
                'index': i,
                'text': "..." + text[-500:] if len(text) > 500 else text,
                'true_label': "positive" if true_label == 1 else "negative", 
                'predicted_label': "positive" if pred_label == 1 else "negative"
            })
    
    return accuracy, api_time, len(sampled_texts), errors

def main():
    # Load IMDB test data
    _, _, test_texts, test_labels = load_imdb_data()
    
    # Sample a consistent test set for both approaches
    # print length of test_texts 
    sample_size = 25000
    indices = random.sample(range(len(test_texts)), min(sample_size, len(test_texts)))
    sampled_texts = [test_texts[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]
    
    print("Evaluating OpenAI API approaches on IMDB sentiment analysis...")
    print("=" * 60)
    print(f"Full test set size: {len(test_texts)}")
    print(f"Using the same {len(sampled_texts)} samples for both approaches")
    
    # Test zero-shot approach with batching
    print("\n1. Zero-shot batch approach:")
    zero_shot_accuracy, zero_shot_time, sample_count, zero_shot_errors = evaluate_openai_approach(
        sampled_texts, sampled_labels, sample_size=len(sampled_texts), use_few_shot=False, batch_size=10
    )
    print(f'Zero-shot Accuracy: {zero_shot_accuracy:.4f}')
    print(f'Zero-shot Time: {zero_shot_time:.2f} seconds for {sample_count} samples')
    print(f'API calls made: ~{sample_count // 10}')
    # Estimate: ~150 tokens per review (500 chars + prompt) * sample_count * $0.4/1M tokens
    estimated_tokens = sample_count * 150
    cost_estimate = (estimated_tokens / 1_000_000) * 0.4
    print(f'Cost estimate: ~${cost_estimate:.3f} (estimated {estimated_tokens:,} tokens)')
    
    # Test few-shot approach with batching
    print("\n2. Few-shot batch approach:")
    few_shot_accuracy, few_shot_time, sample_count, few_shot_errors = evaluate_openai_approach(
        sampled_texts, sampled_labels, sample_size=len(sampled_texts), use_few_shot=True, batch_size=10
    )
    print(f'Few-shot Accuracy: {few_shot_accuracy:.4f}')
    print(f'Few-shot Time: {few_shot_time:.2f} seconds for {sample_count} samples')
    print(f'API calls made: ~{sample_count // 10}')
    # Estimate: ~250 tokens per review (500 chars + longer few-shot prompt) * sample_count * $0.4/1M tokens
    estimated_tokens = sample_count * 250
    cost_estimate = (estimated_tokens / 1_000_000) * 0.4
    print(f'Cost estimate: ~${cost_estimate:.3f} (estimated {estimated_tokens:,} tokens)')
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Zero-shot: {zero_shot_accuracy:.4f} accuracy")
    print(f"Few-shot: {few_shot_accuracy:.4f} accuracy")
    print("Note: Fine-tuned DistilBERT typically achieves ~92.5% accuracy")
    
    # Show first 10 errors for each approach
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS:")
    
    print(f"\nZero-shot errors (showing first 10 of {len(zero_shot_errors)}):")
    for i, error in enumerate(zero_shot_errors[:10]):
        print(f"{i+1}. True: {error['true_label']}, Predicted: {error['predicted_label']}")
        print(f"   Text: {error['text']}")
        print()
    
    print(f"\nFew-shot errors (showing first 10 of {len(few_shot_errors)}):")
    for i, error in enumerate(few_shot_errors[:10]):
        print(f"{i+1}. True: {error['true_label']}, Predicted: {error['predicted_label']}")
        print(f"   Text: {error['text']}")
        print()

if __name__ == "__main__":
    main()