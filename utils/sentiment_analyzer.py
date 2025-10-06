from transformers import pipeline
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    """
    Implements sentiment analysis using Hugging Face transformers
    with improved neutral detection
    """
    
    def __init__(self):
        """Initialize sentiment analysis pipeline"""
        try:
            # Load pre-trained sentiment model (ML framework requirement)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.use_hf = True
            print("✓ Hugging Face sentiment model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load HF model. Using Perplexity API. Error: {e}")
            self.use_hf = False
            self.api_key = os.getenv("PERPLEXITY_API_KEY")
    
    def analyze_sentiment_hf(self, text):
        """
        Analyze sentiment using Hugging Face model with AGGRESSIVE neutral detection
        """
        result = self.sentiment_pipeline(text)[0]
        label = result['label']  # POSITIVE or NEGATIVE
        score = result['score']  # Confidence score
        
        # AGGRESSIVE NEUTRAL DETECTION:
        if score < 0.70:
            sentiment = "neutral"
            confidence = 0.75
        elif label == "POSITIVE":
            sentiment = "positive"
            confidence = score
        else:
            sentiment = "negative"
            confidence = score
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "method": "Hugging Face DistilBERT"
        }

    def analyze_sentiment_perplexity(self, text):
        """
        Analyze sentiment using Perplexity API as backup
        Returns: dict with label and score
        """
        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a sentiment analyzer. Classify the sentiment of the given text as exactly one of: positive, negative, or neutral. Respond with ONLY the sentiment word."
                },
                {
                    "role": "user",
                    "content": f"Classify sentiment: {text}"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 10
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].lower()
            
            if 'positive' in content:
                sentiment = "positive"
            elif 'negative' in content:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": 0.85,
                "method": "Perplexity API"
            }
        
        except Exception as e:
            print(f"Perplexity API error: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "method": "Default (Error)"
            }
    
    def detect_neutral_keywords(self, text):
        """
        Detect neutral/factual indicators with STRONGER detection
        """
        strong_neutral = ['how', 'what', 'when', 'where', 'why', 'explain', 'describe']
        neutral_keywords = [
            'process', 'method', 'system', 'algorithm', 'mechanism', 
            'function', 'works', 'operates', 'consists', 'comprises', 
            'includes', 'definition', 'technical', 'scientific',
            'data', 'information', 'computation', 'analysis'
        ]
        
        text_lower = text.lower()
        
        for keyword in strong_neutral:
            if keyword in text_lower.split():  # check as whole word
                return True
        
        keyword_count = sum(1 for keyword in neutral_keywords if keyword in text_lower)
        return keyword_count >= 2

    def analyze(self, text):
        """
        Main method with PRIORITY on neutral detection
        """
        if not text or len(text.strip()) == 0:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "method": "Empty input"
            }
        
        text_truncated = text[:512]
        
        try:
            has_neutral_keywords = self.detect_neutral_keywords(text_truncated)
            
            if self.use_hf:
                result = self.analyze_sentiment_hf(text_truncated)
                
                if has_neutral_keywords:
                    result['sentiment'] = 'neutral'
                    result['confidence'] = 0.80
                    result['method'] = 'Keyword-Based Neutral Detection + HF'
                
                return result
            else:
                return self.analyze_sentiment_perplexity(text_truncated)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return self.analyze_sentiment_perplexity(text_truncated)


# Test the improved analyzer
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_prompts = [
        "This is absolutely wonderful and amazing!",  # positive
        "This is terrible and disappointing.",       # negative
        "How does artificial intelligence process data?",  # neutral
        "The algorithm consists of several steps.",        # neutral
        "Renewable energy is great for the environment!",  # positive
        "Climate change poses serious threats.",           # negative
        "Machine learning models use neural networks."     # neutral
    ]
    
    print("Testing Sentiment Analyzer:\n")
    for prompt in test_prompts:
        result = analyzer.analyze(prompt)
        print(f"Prompt: {prompt}")
        print(f"Sentiment: {result['sentiment'].upper()} ({result['confidence']:.2%})")
        print(f"Method: {result['method']}\n")
