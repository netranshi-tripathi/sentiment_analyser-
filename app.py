import streamlit as st
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.text_generator import TextGenerator
import time

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Text Generator",
    page_icon="✍️",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_models():
    """Load sentiment analyzer and text generator (cached)"""
    sentiment_analyzer = SentimentAnalyzer()
    text_generator = TextGenerator()
    return sentiment_analyzer, text_generator

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background-color: #C8E6C9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .sentiment-negative {
        background-color: #FFCDD2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
    }
    .sentiment-neutral {
        background-color: #E0E0E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9E9E9E;
    }
    .generated-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #BDBDBD;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown("<div class='main-header'>✍️ AI Sentiment Text Generator</div>", unsafe_allow_html=True)
st.markdown("**Generate paragraphs and essays aligned with the sentiment of your prompt**")

# Load models
try:
    with st.spinner("Loading AI models..."):
        sentiment_analyzer, text_generator = load_models()
    st.success("✓ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    # Manual sentiment override
    use_manual_sentiment = st.checkbox("Override Detected Sentiment", value=False)
    manual_sentiment = None
    if use_manual_sentiment:
        manual_sentiment = st.selectbox(
            "Select Sentiment",
            ["positive", "negative", "neutral"]
        )
    
    # Length control
    length = st.select_slider(
        "Text Length",
        options=["short", "medium", "long"],
        value="medium",
        help="Short: 100-200 words, Medium: 300-500 words, Long: 500-800 words"
    )
    
    # Temperature control
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values = more creative, Lower values = more focused"
    )
    
    st.divider()
    st.markdown("**About**")
    st.info("This app uses Hugging Face DistilBERT for sentiment analysis and Perplexity API for text generation.")

# Main input area
st.subheader("📝 Enter Your Prompt")
user_prompt = st.text_area(
    "Write a topic or prompt...",
    height=150,
    placeholder="Example: Write about the future of artificial intelligence...",
    help="Enter a topic or prompt. The system will detect its sentiment and generate aligned text."
)

# Generate button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    generate_button = st.button("🚀 Generate Text", use_container_width=True, type="primary")

# Generation logic
if generate_button:
    if not user_prompt or len(user_prompt.strip()) < 10:
        st.warning("⚠️ Please enter a prompt with at least 10 characters.")
    else:
        # Step 1: Sentiment Analysis
        st.subheader("🔍 Sentiment Analysis")
        with st.spinner("Analyzing sentiment..."):
            sentiment_result = sentiment_analyzer.analyze(user_prompt)
            time.sleep(0.5)  # UX enhancement
        
        detected_sentiment = sentiment_result['sentiment']
        confidence = sentiment_result['confidence']
        method = sentiment_result['method']
        
        # Display sentiment
        sentiment_class = f"sentiment-{detected_sentiment}"
        sentiment_emoji = {
            "positive": "😊",
            "negative": "😟",
            "neutral": "😐"
        }
        
        st.markdown(f"""
        <div class='{sentiment_class}'>
            <h3>{sentiment_emoji[detected_sentiment]} Detected Sentiment: {detected_sentiment.upper()}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Method:</strong> {method}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use manual override if selected
        final_sentiment = manual_sentiment if use_manual_sentiment else detected_sentiment
        
        if use_manual_sentiment:
            st.info(f"ℹ️ Using manual override: **{final_sentiment}**")
        
        # Step 2: Text Generation
        st.subheader("✨ Generated Text")
        with st.spinner("Generating sentiment-aligned text... This may take 10-20 seconds."):
            generation_result = text_generator.generate(
                sentiment=final_sentiment,
                user_prompt=user_prompt,
                length=length,
                temperature=temperature
            )
        
        # Display results
        if generation_result['success']:
            generated_text = generation_result['generated_text']
            word_count = generation_result['word_count']
            model = generation_result['model']
            
            st.markdown(f"""
            <div class='generated-box'>
                {generated_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", word_count)
            with col2:
                st.metric("Sentiment", final_sentiment.capitalize())
            with col3:
                st.metric("Model", "Perplexity API")
            
            # Citations (if available)
            if generation_result.get('citations'):
                with st.expander("📚 Sources & Citations"):
                    for i, citation in enumerate(generation_result['citations'], 1):
                        st.write(f"{i}. {citation}")
            
            # Download option
            st.download_button(
                label="📥 Download Generated Text",
                data=generated_text,
                file_name=f"generated_text_{final_sentiment}.txt",
                mime="text/plain"
            )
        else:
            st.error(f"❌ Generation failed: {generation_result['error']}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #757575;'>
    <p>Built with Streamlit | Powered by Perplexity API & Hugging Face</p>
</div>
""", unsafe_allow_html=True)
