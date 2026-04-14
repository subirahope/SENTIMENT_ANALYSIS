"""
Automated Text Mining Solution for Real-Time Sentiment Analysis
Fixed Dashboard Design - Better Visibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from collections import Counter
from wordcloud import WordCloud
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ============================================================================
# LAZY LOADING FOR NLTK
# ============================================================================

_nltk_loaded = False

def load_nltk():
    global _nltk_loaded
    if not _nltk_loaded:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        _nltk_loaded = True
    return nltk

# ============================================================================
# TEXT PREPROCESSOR CLASS
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.nltk = None
        self.custom_stopwords = {
            'na', 'ya', 'wa', 'ni', 'cha', 'vya', 'kwa', 'kwenye', 'katika',
            'kuwa', 'kuna', 'kutoka', 'kama', 'hii', 'hizi', 'hicho', 'ile',
            'hao', 'hawa', 'wale', 'yule', 'zile', 'kwamba', 'ingawa', 'basi',
            'ndio', 'ndiyo', 'hivyo', 'hapo', 'huko', 'huku', 'pale', 'kule',
            'nyuma', 'mbele', 'juu', 'chini', 'ndani', 'nje', 'baada', 'kabla',
            'wakati', 'muda', 'sasa', 'bado', 'tena', 'zaidi', 'pia', 'sana',
            'kidogo', 'tu', 'hata', 'mara', 'za', 'la', 'ali', 'ana', 'nil',
            'tuli', 'mli', 'wali', 'nina', 'una', 'ana', 'tuna', 'mna', 'wana',
            'nime', 'ume', 'ame', 'tume', 'mme', 'wame', 'si', 'hu', 'ki', 'vi',
            'the', 'a', 'an', 'and', 'of', 'to', 'is', 'in', 'it', 'that',
            'for', 'on', 'with', 'as', 'by', 'at', 'from', 'or', 'but'
        }
        
        self.sheng_map = {
            'bie': 'nzuri', 'ngori': 'ngumu', 'kali': 'nzuri', 'poa': 'nzuri',
            'freshi': 'nzuri', 'bora': 'nzuri', 'choma': 'mbaya', 'takataka': 'mbaya',
            'kubaya': 'mbaya', 'kibaya': 'mbaya', 'baya': 'mbaya', 'pumbavu': 'jinga',
            'kijinga': 'jinga', 'ujinga': 'jinga', 'fiti': 'nzuri', 'safi': 'nzuri',
            'moto': 'nzuri', 'dah': 'sana', 'kwelikweli': 'sana', 'haki': 'kweli',
            'jameni': 'tafadhali', 'wacha': 'acha'
        }
        
        self.negations = {'sio', 'si', 'haku', 'hau', 'ha', 'siwezi', 'sik', 
                         'bila', 'kutokuwa', 'no', 'not', 'never', 'none'}
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def handle_sheng_slang(self, text):
        words = text.split()
        normalized = [self.sheng_map.get(word, word) for word in words]
        return ' '.join(normalized)
    
    def remove_stopwords(self, text):
        words = text.split()
        filtered = [w for w in words if w not in self.custom_stopwords or w in self.negations]
        return ' '.join(filtered)
    
    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        text = self.clean_text(text)
        text = self.handle_sheng_slang(text)
        text = self.remove_stopwords(text)
        return text
    
    def batch_preprocess(self, texts):
        return [self.preprocess(text) for text in texts]

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_sentiment_pie_chart(sentiment_counts):
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        hole=0.4,
        marker_colors=['#2ecc71', '#e74c3c', '#95a5a6'],
        textinfo='percent+label',
        textposition='auto'
    )])
    fig.update_layout(
        title="Sentiment Distribution",
        height=450,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_sentiment_bar_chart(sentiment_counts):
    fig = go.Figure(data=[go.Bar(
        x=list(sentiment_counts.keys()),
        y=list(sentiment_counts.values()),
        marker_color=['#2ecc71', '#e74c3c', '#95a5a6'],
        text=list(sentiment_counts.values()),
        textposition='auto'
    )])
    fig.update_layout(
        title="Sentiment Distribution (Bar Chart)",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_word_cloud(texts, title="Word Cloud"):
    if not texts:
        return None
    all_text = ' '.join(texts)
    wordcloud = WordCloud(
        width=700, height=350,
        background_color='white',
        colormap='viridis',
        max_words=80
    ).generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_confidence_gauge(confidence_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score * 100,
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2c3e50"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f8d7da'},
                {'range': [50, 75], 'color': '#fff3cd'},
                {'range': [75, 100], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_model_comparison_chart(model_scores):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig = go.Figure()
    colors = ['#3498db', '#e74c3c']
    for i, (model_name, scores) in enumerate(model_scores.items()):
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=[scores.get(m.lower(), 0) for m in metrics],
            text=[f'{scores.get(m.lower(), 0):.3f}' for m in metrics],
            textposition='auto',
            marker_color=colors[i % len(colors)]
        ))
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        height=450,
        yaxis_range=[0, 1],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_top_words_chart(word_frequencies, title="Most Frequent Words", top_n=12):
    top_words = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n])
    fig = go.Figure(data=[go.Bar(
        x=list(top_words.values()),
        y=list(top_words.keys()),
        orientation='h',
        marker_color='#3498db',
        text=list(top_words.values()),
        textposition='outside'
    )])
    fig.update_layout(
        title=title,
        xaxis_title="Frequency",
        yaxis_title="Word",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with visible metric cards
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header h3 {
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    
    /* Metric Cards - VISIBLE */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #e0e0e0;
        margin: 0.5rem;
    }
    .metric-card h3 {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    .metric-card-total h2 { color: #3498db; }
    .metric-card-positive h2 { color: #27ae60; }
    .metric-card-negative h2 { color: #e74c3c; }
    
    /* Sentiment boxes */
    .sentiment-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
        color: #383d41;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #6c757d;
        margin: 0.5rem 0;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%);
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    
    /* Main content background */
    .main > div {
        background-color: #f0f2f6;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Header
st.markdown("""
<div class="main-header">
    <h1> Automated Text Mining Solution</h1>
    <h3>Real-Time Sentiment Analysis for Customer Feedback</h3>
    <p style="margin-top: 0.5rem; opacity: 0.9;">Supporting English, Swahili, and Sheng code-switching</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# Navigation")
    page = st.radio(
        "Select Page",
        [" Dashboard", " Model Training", " Real-Time Analysis", " Analytics", " About"],
        format_func=lambda x: x.strip()
    )
    
    st.markdown("---")
    st.markdown("### Dataset Status")
    if st.session_state.data_loaded:
        st.success(f"✓ Loaded: {len(st.session_state.df)} records")
        if 'labels' in st.session_state.df.columns:
            pos = len(st.session_state.df[st.session_state.df['labels'] == 'positive'])
            neg = len(st.session_state.df[st.session_state.df['labels'] == 'negative'])
            st.metric("Positive", pos, delta=None)
            st.metric("Negative", neg, delta=None)
    else:
        st.info("No data loaded")
    
    st.markdown("---")
    st.markdown("### Team Members")
    st.markdown("""
    - Timothy Joseph
    - Mukiri Sharon
    - Caleb Ngumbau
    - Osama Mohammed
    - Monicah Gitahi
    """)
    st.markdown("---")
    st.markdown("**Supervisor:** Mr. Vincent Mwai")
    st.markdown("**JKUAT - 2026**")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if page == " Dashboard":
    st.header("Dashboard Overview")
    
    if st.session_state.data_loaded:
        # Metrics Row - Visible Cards
        col1, col2, col3 = st.columns(3)
        
        total = len(st.session_state.df)
        pos_count = len(st.session_state.df[st.session_state.df['labels'] == 'positive'])
        neg_count = len(st.session_state.df[st.session_state.df['labels'] == 'negative'])
        pos_pct = (pos_count / total * 100) if total > 0 else 0
        neg_pct = (neg_count / total * 100) if total > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Reviews</h3>
                <h2>{total}</h2>
                <p style="color: #6c757d; margin-top: 0.5rem;">100%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Positive Reviews</h3>
                <h2 style="color: #27ae60;">{pos_count}</h2>
                <p style="color: #27ae60;">{pos_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Negative Reviews</h3>
                <h2 style="color: #e74c3c;">{neg_count}</h2>
                <p style="color: #e74c3c;">{neg_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        sentiment_counts = st.session_state.df['labels'].value_counts().to_dict()
        
        with col1:
            fig_pie = create_sentiment_pie_chart(sentiment_counts)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = create_sentiment_bar_chart(sentiment_counts)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Sample Reviews
        st.markdown("### Sample Customer Reviews")
        
        # Add filter for sample reviews
        filter_option = st.selectbox("Filter by sentiment:", ["All", "Positive", "Negative"])
        
        if filter_option == "Positive":
            sample_df = st.session_state.df[st.session_state.df['labels'] == 'positive'].head(10)
        elif filter_option == "Negative":
            sample_df = st.session_state.df[st.session_state.df['labels'] == 'negative'].head(10)
        else:
            sample_df = st.session_state.df.head(10)
        
        for idx, row in sample_df.iterrows():
            if row['labels'] == 'positive':
                st.markdown(f"""
                <div class="sentiment-positive">
                    <strong>POSITIVE</strong><br>
                    {row['text'][:250]}...
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sentiment-negative">
                    <strong>NEGATIVE</strong><br>
                    {row['text'][:250]}...
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📁 No data loaded. Please go to **Model Training** page to upload your dataset.")
        st.markdown("""
        ### Quick Start
        1. Go to **Model Training** page
        2. Upload your CSV file (must have 'text' and 'labels' columns)
        3. Click **Train Model**
        4. Return here to see the dashboard
        """)

# ============================================================================
# MODEL TRAINING PAGE
# ============================================================================

elif page == " Model Training":
    st.header("Model Training")
    
    st.markdown("""
    Train machine learning models to classify customer feedback sentiment.
    The system supports English, Swahili, and Sheng (code-switching).
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file with customer reviews", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Clean columns
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            if 'labels' in df.columns:
                label_map = {'positive': 1, 'negative': 0}
                df['label_encoded'] = df['labels'].map(label_map)
                df = df.dropna(subset=['label_encoded'])
                df['label_encoded'] = df['label_encoded'].astype(int)
                
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Successfully loaded {len(df)} reviews!")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df[['text', 'labels']].head(10), use_container_width=True)
                
                # Show statistics
                st.subheader("Dataset Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Reviews", len(df))
                with col2:
                    st.metric("Positive Reviews", len(df[df['labels'] == 'positive']))
                with col1:
                    st.metric("Negative Reviews", len(df[df['labels'] == 'negative']))
            else:
                st.error("CSV file must contain a 'labels' column with 'positive' or 'negative' values")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Select Model", ["Naive Bayes", "SVM", "Both"])
        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model... Please wait."):
                
                # Initialize preprocessor
                if st.session_state.preprocessor is None:
                    st.session_state.preprocessor = TextPreprocessor()
                
                preprocessor = st.session_state.preprocessor
                texts = st.session_state.df['text'].astype(str).tolist()
                
                # Preprocess
                with st.spinner("Preprocessing text..."):
                    processed_texts = preprocessor.batch_preprocess(texts)
                
                # Vectorize
                with st.spinner("Vectorizing text..."):
                    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(processed_texts)
                    y = st.session_state.df['label_encoded'].values
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                results = {}
                
                # Train Naive Bayes
                if model_type in ["Naive Bayes", "Both"]:
                    with st.spinner("Training Naive Bayes..."):
                        nb = MultinomialNB()
                        nb.fit(X_train, y_train)
                        y_pred = nb.predict(X_test)
                        results['Naive Bayes'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0),
                            'model': nb
                        }
                
                # Train SVM
                if model_type in ["SVM", "Both"]:
                    with st.spinner("Training SVM..."):
                        svm = SVC(kernel='linear', probability=True, random_state=42)
                        svm.fit(X_train, y_train)
                        y_pred = svm.predict(X_test)
                        results['SVM'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0),
                            'model': svm
                        }
                
                # Save best model
                best_model = max(results, key=lambda x: results[x]['accuracy'])
                st.session_state.model = results[best_model]['model']
                st.session_state.vectorizer = vectorizer
                
                # Save to disk
                joblib.dump(st.session_state.model, 'sentiment_model.pkl')
                joblib.dump(st.session_state.vectorizer, 'vectorizer.pkl')
                
                st.success(f"✅ Training complete! Best model: **{best_model}**")
                st.metric("Accuracy", f"{results[best_model]['accuracy']:.4f}")
                
                # Show results
                st.subheader("Model Performance")
                
                model_scores = {name: {k: v for k, v in m.items() if k != 'model'} 
                               for name, m in results.items()}
                st.plotly_chart(create_model_comparison_chart(model_scores), use_container_width=True)
                
                # Detailed metrics
                st.subheader("Detailed Metrics")
                for name, m in results.items():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"{name} - Accuracy", f"{m['accuracy']:.3f}")
                    with col2:
                        st.metric(f"{name} - Precision", f"{m['precision']:.3f}")
                    with col3:
                        st.metric(f"{name} - Recall", f"{m['recall']:.3f}")
                    with col4:
                        st.metric(f"{name} - F1 Score", f"{m['f1']:.3f}")

# ============================================================================
# REAL-TIME ANALYSIS PAGE
# ============================================================================

elif page == " Real-Time Analysis":
    st.header("Real-Time Sentiment Analysis")
    
    st.markdown("Enter customer feedback below to analyze its sentiment in real-time.")
    
    # Example buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Load Positive Example", use_container_width=True):
            st.session_state.example_text = "Bidhaa hii ni nzuri sana, nimefurahishwa na huduma yenu"
    with col2:
        if st.button("📝 Load Negative Example", use_container_width=True):
            st.session_state.example_text = "Bidhaa hii ni mbaya sana, sijafurahishwa kabisa"
    
    user_input = st.text_area(
        "Customer Feedback:",
        value=st.session_state.get('example_text', ''),
        height=120,
        placeholder="Example: Bidhaa hii ni nzuri sana, nitaipenda..."
    )
    
    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if user_input:
            if st.session_state.model is not None and st.session_state.vectorizer is not None:
                with st.spinner("Analyzing..."):
                    if st.session_state.preprocessor is None:
                        st.session_state.preprocessor = TextPreprocessor()
                    
                    processed = st.session_state.preprocessor.preprocess(user_input)
                    vectorized = st.session_state.vectorizer.transform([processed])
                    prediction = st.session_state.model.predict(vectorized)[0]
                    
                    if hasattr(st.session_state.model, 'predict_proba'):
                        confidence = max(st.session_state.model.predict_proba(vectorized)[0])
                    else:
                        confidence = 0.85
                    
                    sentiment = "positive" if prediction == 1 else "negative"
                    
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "positive":
                            st.markdown("""
                            <div class="sentiment-positive" style="padding: 2rem;">
                                <h2 style="margin: 0;">😊 POSITIVE</h2>
                                <p style="margin-top: 0.5rem;">The customer expressed satisfaction with the product or service.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="sentiment-negative" style="padding: 2rem;">
                                <h2 style="margin: 0;">😞 NEGATIVE</h2>
                                <p style="margin-top: 0.5rem;">The customer expressed dissatisfaction with the product or service.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        fig_gauge = create_confidence_gauge(confidence)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with st.expander("View Analysis Details"):
                        st.write("**Original Text:**", user_input)
                        st.write("**Preprocessed Text:**", processed)
                        st.write("**Prediction Confidence:**", f"{confidence:.2%}")
            else:
                st.error("⚠️ No trained model found. Please go to Model Training page and train a model first.")
        else:
            st.warning("Please enter some text to analyze.")

# ============================================================================
# ANALYTICS PAGE
# ============================================================================

elif page == " Analytics":
    st.header("Advanced Analytics")
    
    if st.session_state.data_loaded:
        # Word Clouds
        st.subheader("Word Cloud Analysis")
        st.markdown("Visualizing the most common words in positive and negative reviews.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Reviews")
            pos_texts = st.session_state.df[st.session_state.df['labels'] == 'positive']['text'].tolist()
            if pos_texts:
                fig = create_word_cloud(pos_texts, "Positive Reviews Word Cloud")
                if fig:
                    st.pyplot(fig)
            else:
                st.info("No positive reviews found")
        
        with col2:
            st.markdown("#### Negative Reviews")
            neg_texts = st.session_state.df[st.session_state.df['labels'] == 'negative']['text'].tolist()
            if neg_texts:
                fig = create_word_cloud(neg_texts, "Negative Reviews Word Cloud")
                if fig:
                    st.pyplot(fig)
            else:
                st.info("No negative reviews found")
        
        # Most Frequent Words
        st.subheader("Most Frequent Words Analysis")
        
        if st.session_state.preprocessor is None:
            st.session_state.preprocessor = TextPreprocessor()
        
        all_texts = st.session_state.df['text'].astype(str).tolist()
        processed = st.session_state.preprocessor.batch_preprocess(all_texts)
        all_words = ' '.join(processed).split()
        word_freq = Counter(all_words)
        
        fig_top = create_top_words_chart(word_freq, "Most Frequent Words Across All Reviews")
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Export Report
        st.markdown("---")
        st.subheader("Export Analysis Report")
        
        if st.button("📥 Generate Report", use_container_width=True):
            total = len(st.session_state.df)
            pos = len(st.session_state.df[st.session_state.df['labels'] == 'positive'])
            neg = len(st.session_state.df[st.session_state.df['labels'] == 'negative'])
            
            report_data = {
                'metric': ['Total Reviews', 'Positive Reviews', 'Negative Reviews', 
                          'Positive Percentage', 'Negative Percentage'],
                'value': [total, pos, neg, f"{(pos/total*100):.1f}%", f"{(neg/total*100):.1f}%"]
            }
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name="sentiment_analysis_report.csv",
                mime="text/csv"
            )
    else:
        st.info("📁 No data loaded. Please go to Model Training page to load and analyze data.")

# ============================================================================
# ABOUT PAGE
# ============================================================================

else:
    st.header("About This Project")
    
    st.markdown("""
    ### Project Overview
    
    This automated text mining solution performs real-time sentiment analysis on customer feedback,
    specifically tailored for the Kenyan retail market. The system handles code-switching between
    English, Swahili, and Sheng.
    
    ---
    
    ### Problem Statement
    
    Retailers face significant challenges in manually analyzing large volumes of customer feedback.
    Existing automated tools fail to interpret the linguistic nuances of the Kenyan market,
    where code-switching and Sheng are common.
    
    ---
    
    ### Objectives
    
    | # | Objective |
    |---|-----------|
    | 1 | Curate, preprocess, and label a comprehensive dataset of retail customer feedback |
    | 2 | Engineer and select linguistic features for sentiment analysis |
    | 3 | Train and optimize multiple machine learning classification models |
    | 4 | Develop a functional web-based application |
    | 5 | Validate the system's performance and business utility |
    
    ---
    
    ### Methodology (CRISP-DM)
    
    The project follows the Cross-Industry Standard Process for Data Mining:
    
    - **Business Understanding** - Define project objectives
    - **Data Understanding** - Collect and explore data
    - **Data Preparation** - Clean and transform data
    - **Modeling** - Train ML models (Naive Bayes, SVM)
    - **Evaluation** - Assess model performance
    - **Deployment** - Deploy web application
    
    ---
    
    ### Technologies Used
    
    | Technology | Purpose |
    |------------|---------|
    | Python 3.10+ | Core programming language |
    | Streamlit | Web application framework |
    | Scikit-learn | Machine learning algorithms |
    | NLTK | Natural language processing |
    | Pandas | Data manipulation |
    | Plotly/Matplotlib | Data visualization |
    | WordCloud | Text visualization |
    
    ---
    
    ### Team Members
    
    | Name | Registration Number |
    |------|---------------------|
    | Timothy Joseph | SCM211-1237/2022 |
    | Mukiri Sharon Wanjiku | SCM211-1372/2021 |
    | Caleb Ngumbau Paul | SCM211-0282/2021 |
    | Osama Abdullahi Mohammed | SCM211-0310/2021 |
    | Monicah Wanja Gitahi | SCM211-1329/2021 |
    
    ---
    
    ### Supervision
    
    **Supervisor:** Mr. Vincent Mwai
    
    ### Institution
    
    **Jomo Kenyatta University of Agriculture and Technology**  
    Department of Pure and Applied Mathematics  
    School of Mathematical Sciences
    
    ### Year
    
    **2026**
    """)
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("Current Dataset Statistics")
        
        total = len(st.session_state.df)
        pos = len(st.session_state.df[st.session_state.df['labels'] == 'positive'])
        neg = len(st.session_state.df[st.session_state.df['labels'] == 'negative'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", total)
        with col2:
            st.metric("Positive Reviews", pos)
        with col3:
            st.metric("Negative Reviews", neg)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; padding: 1rem;'>"
    "© 2026 - Automated Text Mining Solution for Real-Time Sentiment Analysis | JKUAT"
    "</div>",
    unsafe_allow_html=True
)