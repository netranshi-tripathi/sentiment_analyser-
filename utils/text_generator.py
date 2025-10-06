import requests
import os
from dotenv import load_dotenv

load_dotenv()

class TextGenerator:
    """
    Generates sentiment-aligned text using Perplexity API
    """

    def __init__(self):
        """Initialize Perplexity API client"""
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"

        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")

        print("✓ Perplexity API configured successfully")

    def create_sentiment_prompt(self, sentiment, user_prompt, length="medium"):
        """
        Create sentiment-conditioned prompt
        Args:
            sentiment: positive, negative, or neutral
            user_prompt: user's input topic
            length: short, medium, or long
        Returns:
            complete prompt string
        """
        length_map = {
            "short": "100-200 words",
            "medium": "300-500 words",
            "long": "500-800 words"
        }

        word_count = length_map.get(length, "300-500 words")

        sentiment_instructions = {
            "positive": "Write in an optimistic, uplifting, and encouraging tone. Highlight positive aspects, benefits, and hopeful perspectives.",
            "negative": "Write in a critical, cautionary, or pessimistic tone. Focus on challenges, drawbacks, and concerning aspects.",
            "neutral": "Write in an objective, balanced, and informative tone. Present facts without emotional bias."
        }

        instruction = sentiment_instructions.get(sentiment, sentiment_instructions["neutral"])

        # Create comprehensive prompt
        final_prompt = (
            f"{instruction}\n\n"
            f"Write a well-structured {sentiment} paragraph or essay of approximately {word_count} "
            f"about the following topic:\n\n{user_prompt.strip()}\n\n"
            f"Ensure the content is coherent, engaging, and maintains the {sentiment} sentiment throughout."
        )

        return final_prompt

    def generate(self, sentiment, user_prompt, length="medium", temperature=0.7):
        """
        Generate sentiment-aligned text using Perplexity API
        Args:
            sentiment: positive, negative, or neutral
            user_prompt: user's topic
            length: short, medium, or long
            temperature: creativity (0.0-1.0)
        Returns:
            dict with generated text or error
        """
        # Validate input
        if not user_prompt or len(user_prompt.strip()) < 5:
            return {
                "success": False,
                "error": "Input prompt is too short or empty.",
                "generated_text": None
            }

        # Create prompt
        final_prompt = self.create_sentiment_prompt(sentiment, user_prompt, length)

        # Token limits
        token_map = {
            "short": 300,
            "medium": 600,
            "long": 1000
        }
        max_tokens = token_map.get(length, 600)

        # UPDATED: Use valid Perplexity API model names (as of 2025)
        # Valid models: "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", "sonar-deep-research"
        payload = {
            "model": "sonar-pro",  # Use sonar-pro for better quality answers
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        print("\n=== Perplexity API Request ===")
        print(f"Model: {payload['model']}")
        print(f"Sentiment: {sentiment}")
        print(f"Length: {length}")
        print(f"Max Tokens: {max_tokens}")
        print(f"Prompt preview: {final_prompt[:150]}...")
        print("==============================\n")

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            print(f"Response Status: {response.status_code}")

            response.raise_for_status()

            result = response.json()
            generated_text = result['choices'][0]['message']['content']

            # Get citations if available
            citations = result.get('citations', [])

            return {
                "success": True,
                "generated_text": generated_text,
                "sentiment": sentiment,
                "length": length,
                "model": payload["model"],
                "citations": citations,
                "word_count": len(generated_text.split())
            }

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            error_detail = ""

            try:
                error_detail = e.response.json()
                print(f"API Error Details: {error_detail}")
            except:
                error_detail = e.response.text
                print(f"API Error Text: {error_detail}")

            if code == 400:
                error_msg = f"Bad request. Details: {error_detail}"
            elif code == 401:
                error_msg = "Invalid API key. Check your credentials."
            elif code == 402:
                error_msg = "Insufficient credits. Please add credits to your Perplexity account."
            elif code == 429:
                error_msg = "Rate limit exceeded. Wait before retrying."
            else:
                error_msg = f"API Error {code}: {error_detail}"

            return {
                "success": False,
                "error": error_msg,
                "generated_text": None
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. Try again.",
                "generated_text": None
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "generated_text": None
            }


# Test the generator
if __name__ == "__main__":
    print("Testing Perplexity Text Generator...")
    gen = TextGenerator()
    
    test_prompt = "artificial intelligence in healthcare"
    result = gen.generate("positive", test_prompt, "medium", 0.7)
    
    if result["success"]:
        print("\n✓ Generation Successful!")
        print(f"\nGenerated Text ({result['word_count']} words):")
        print(result['generated_text'])
        if result['citations']:
            print(f"\nCitations: {result['citations']}")
    else:
        print(f"\n✗ Generation Failed: {result['error']}")
