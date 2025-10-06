# AI Sentiment Text Generator

## Overview
An AI-powered text generation system that analyzes sentiment from user prompts and generates sentiment-aligned paragraphs or essays.

## Features
- ✅ Sentiment analysis using Hugging Face DistilBERT
- ✅ Text generation using Perplexity API
- ✅ Interactive Streamlit frontend
- ✅ Manual sentiment override
- ✅ Adjustable text length and creativity
- ✅ Real-time processing

## Technology Stack
- **Python 3.8+**
- **Sentiment Analysis:** Hugging Face Transformers (DistilBERT)
- **Text Generation:** Perplexity API (LLaMA 3.1 Sonar)
- **Frontend:** Streamlit
- **ML Frameworks:** PyTorch, Transformers

## Installation

1. Clone repository:
git clone <your-repo>
cd sentiment-text-generator


2. Install dependencies:
pip install -r requirements.txt


3. Configure API key:
Create `.env` file:
PERPLEXITY_API_KEY=your_key_here


4. Run application:
streamlit run app.py


## Methodology

### Sentiment Analysis
Uses pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face. Classifies text into positive, negative, or neutral with confidence scores.

### Text Generation
Leverages Perplexity API with sentiment-conditioned prompts. Uses LLaMA 3.1 Sonar Large model for high-quality, coherent text generation.

### Pipeline
1. User inputs prompt
2. Sentiment analysis detects emotion
3. Sentiment conditions generation prompt
4. Perplexity API generates aligned text
5. Results displayed with metadata

## Dataset
- Pre-trained models used (no custom training required)
- DistilBERT trained on SST-2 sentiment dataset
- LLaMA 3.1 trained on diverse web corpus

## Challenges & Solutions
1. **API Rate Limits:** Implemented error handling and user feedback
2. **Sentiment Alignment:** Created sentiment-specific prompt templates
3. **Response Time:** Added loading indicators for better UX
4. **Model Loading:** Cached models using Streamlit for faster startup

## Future Improvements
- Multi-language support
- Fine-tuning on domain-specific data
- Batch processing capability
- Export to multiple formats

