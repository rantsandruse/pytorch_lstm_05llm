# Pytorch with NLP in five days - Day 5: Parameter-Efficient Fine-tuning with LoRA

Note: run `uv sync` to install related packages 

Ironically, it took me another four years to finally write this final post of the series. The past three years have ushered in a brave new world of large language models, both wondrous and unpredictable. Sentiment analysis is no longer treated as a siloed NLP task, as LLMs can multitask with ease. Hand-coding models from scratch is no longer a necessity for most small companies either, as there's no need to reinvent a worse wheel at a higher cost when OPENAI API is a call away. 

While the argument is not yet settle for customized, small LLMs vs. generalized, large LLM, I will close out this series by exploring: 
1. **PEFT(Parameter efficient fine tuning) using LoRA (Low-Rank Adaptation)** This technique allows us to achieve comparable performance to full fine-tuning while using a fraction of the memory and computational resources.
2. **Just calling OPENAI API** And see how the results stack up. 

### The Parameter-Efficient Revolution

Full fine-tuning of large models requires updating and storing gradients for millions of parameters, making it memory-intensive and slow. LoRA (Low-Rank Adaptation) learns small, low-rank matrices that adapt the pre-trained model's behavior without modifying the original weights. This approach typically uses 90% less memory while maintaining similar performance. 

There's also a practical reason I chose this path. Hugging Face already provides an excellent full fine-tuning tutorial for the IMDB dataset using DistilBERT ([link here](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)). Rather than repeating that, I wanted to highlight an alternative approach—one that's both efficient and increasingly relevant in the LLM era.


### What is LoRA (Low-Rank Adaptation)?

LoRA works by decomposing the weight updates into two smaller matrices (A and B) such that:
- Original weight: W₀ + ΔW = W₀ + BA
- Where A is r×d and B is d×r, with r << d (rank is much smaller than the original dimension)
- Only A and B are trained, keeping W₀ frozen
- The original weight ΔW is dxd per layer, whereas the lora parameters are 2xdxr per layer, so reduction is rougly 2xr/d (e.g. d = 768 and r = 8, then reduction = 12,288/589,824 = 2.1%)

This dramatically reduces the number of trainable parameters while maintaining model expressivity.

#### Setting Up LoRA Fine-tuning with PEFT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    r=8,                         # Rank - smaller = more efficient but less expressive
    lora_alpha=16,               # Scaling parameter
    lora_dropout=0.1,            # Dropout for LoRA layers
    target_modules=['query', 'value']  # Apply LoRA to attention matrices
)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

#### Result Summary

```
🚀 Starting LoRA Fine-tuning on Google Colab!
============================================================
Using device: cuda
Loading IMDB dataset (small subset for Colab)...
Training samples: 25000
Test samples: 25000
Setting up LoRA model for Colab...

Base model parameters: 82,119,938
trainable params: 739,586 || all params: 82,859,524 || trainable%: 0.8926

Training Configuration:
- Epochs: 2
- Batch size: 8
- Learning rate: 2e-4
- Total steps: 6250

🎯 Starting training on 25000 samples...

📚 Epoch 1/2
📊 Epoch 1 Results:
   Train Loss: 0.2963
   Train Accuracy: 0.9220
   Test Accuracy: 0.9175

📚 Epoch 2/2
📊 Epoch 2 Results:
   Train Loss: 0.2470
   Train Accuracy: 0.9268
   Test Accuracy: 0.9205

⏱️ Total training time: 1662.1 seconds (27.7 minutes)

==================================================
PARAMETER COMPARISON DEMO
==================================================
Full Fine-tuning Parameters: 82,000,000
LoRA Parameters: 739,586
Parameter Reduction: 99.1%
Memory Savings: ~79%

🎉 TRAINING COMPLETE!🎉
============================================================
Final Test Accuracy: 92.1%
Parameter Reduction: 99.1%
Training Time: 27.7 minutes

💡 Key Takeaways:
   • LoRA achieves great performance with 99%+ fewer parameters
   • Perfect for fine-tuning on limited hardware like Colab
   • Much faster than full fine-tuning

✅ Training completed! Final accuracy: 92.1%
```

#### Running IMDB dataset through OPENAI API 
Out of curiousity, I also ran IMDB test set through OpenAI API. Note that I'm batching 10 samples at a time to reduce overhead, and also only taking the last 500 hundred characters to reduce the amount of tokens. 

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def batch_classify_sentiment(texts, batch_size=10, model="gpt-4.1-mini"):
    """Classify sentiment using OpenAI API with batch processing"""
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Create numbered list of reviews (using last 500 chars for better sentiment detection)
        reviews_list = "\n".join([f"{j+1}. {text[:500]}" for j, text in enumerate(batch_texts)])
        
        prompt = f"""You are a sentiment analysis expert. Classify each movie review as positive or negative sentiment.

Reviews:
{reviews_list}

Respond with exactly one number per review (1 for positive, 0 for negative), separated by commas. Example: 1, 0, 1

Classifications:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=len(batch_texts) * 5,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        predictions = [int(pred.strip()) if pred.strip() in ['0', '1'] else 0 
                      for pred in result.split(",")]
        all_predictions.extend(predictions)
    
    return all_predictions
```

And here are the comparisons: 
- LSTM (from tutorial 4): Accuracy ~ 0.88 
- distillRoberta-base + LORA: Accuracy ~ 0.92 
- OPENAI GPT4o: ~ 0.91 for zeroshot; 0.93 for few shot. 

Arguably comparing Roberta vs OPENAI model is a bit of a David vs Goliath comparison. DistillRoberta-base has ~82M parameters vs ~200B parameters for GPT4. Yet for sentiment analysis, smaller model proves to perform well enough,  even though the larger models has a slight advantage.  

#### When to Use LoRA?

- **Limited computational resources** (single GPU, limited VRAM)
- **Multiple similar tasks** where you want to share a base model
- **Rapid experimentation** with different model configurations  
- **Production environments** where model size matters
- **Fine-tuning very large models** (>1B parameters)

### Conclusion
This concludes our "PyTorch with NLP in Five Days" journey, which took me four years to finish (and end up scrambling to catch up). We've evolved from data processing for basic neural networks training (Day 1) to parameter-efficient transformer fine-tuning and calling commerical LLM API (Day 5). Day1-Day4 code was entirely handwritten, whereas Day5 was ~99% vibe-coded. 

This may not be a bold prediction, but I believe that large model capabilities will continue to push the boundaries of reasoning, creativity and multi-tasking, while small, on-device models are becoming increasingly powerful for specific tasks, such as sentiment analysis, personalization and privacy-sensitive applications.  

## Tips and Tricks 

### LoRA Fine-tuning
**Tune the rank parameter (r)**: The rank parameter is crucial for balancing efficiency and performance. Start with r=8 or r=16 and experiment. Higher ranks give more expressivity but use more parameters. Although I didn't fully explore this due to Colab resource limitations.

### OpenAI API Optimization
**Batch processing**: I batched the reviews to save on overhead costs when making API calls.

**Token optimization**: I used only the last 500 characters of each review to save tokens. This came from data analysis - I noticed that reviews often start with misleadingly positive notes but end with the actual sentiment (e.g., "this is my favorite childhood movie... but now I hate it so much"). I guess the more important tip here is **always examine your data first**. This is a technique that will hopefully never go out of date, even though the best models surely will. 

**Prompt tuning**: Few shot clearly outperforms zero-shot here. Most likely with some additonal error examples and prompt tuning, we could achieve even higher accuracy. 



## files
- **`fine_tuning_lora_colab.ipynb`**: Google Colab notebook with optimized LoRA implementation, ready to run on free Colab tier
- **`openai_api_run.ipynb`**: Notebook for running IMDB sentiment analysis using OpenAI API for comparison
- **`out_of_box.py`**: Python script for running GPT4 model out of the box without fine-tuning
- **`out_of_box.log`**: Log file containing results from the out-of-box model runs


