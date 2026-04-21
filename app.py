"""
Automated Text Mining Solution for Real-Time Sentiment Analysis
Optimized Streamlit Dashboard - Compatible with KE_Retail_Sentiment_Dataset.xlsx
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
from plotly.subplots import make_subplots

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
#=============================================================================
# Add this to your app.py right after loading the data

st.subheader("Data Diagnostic - Why 100% Accuracy?")

# 1. Check for duplicates
duplicates = df_binary.duplicated(subset=['Review_Text']).sum()
st.write(f"**Duplicate reviews:** {duplicates}")

# 2. Check if sentiment appears in text
contains_label = df_binary['Review_Text'].str.lower().str.contains('positive|negative|neutral').sum()
st.write(f"**Reviews containing sentiment words:** {contains_label}")

# 3. Show sample of training vs test overlap risk
st.write("**Sample reviews from dataset:**")
for i, row in df_binary.head(5).iterrows():
    st.write(f"Text: {row['Review_Text'][:100]}...")
    st.write(f"Label: {row['Sentiment_Label']}")
    st.write("---")

# 4. Check if Star_Rating perfectly predicts sentiment (if column exists)
if 'Star_Rating' in df.columns:
    rating_sentiment = df.groupby('Star_Rating')['Sentiment_Label'].value_counts()
    st.write("**Star Rating vs Sentiment:**")
    st.dataframe(rating_sentiment)

#=============================================================================

# ============================================================================
# TEXT PREPROCESSOR CLASS (UPDATED FOR KENYAN DATASET)
# ============================================================================

class TextPreprocessor:
    """Fast text preprocessor for Swahili/English/Sheng text"""
    
    def __init__(self):
        self.nltk = None
        # Custom stopwords combining English and Swahili
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
            'for', 'on', 'with', 'as', 'by', 'at', 'from', 'or', 'but', 'for',
            'of', 'with', 'without', 'after', 'before', 'up', 'down', 'into'
        }
        
        # Sheng slang mapping (based on your dataset's Annotation Guide)
        self.sheng_map = {
            # Positive Sheng markers
            'bie': 'nzuri', 'kali': 'nzuri', 'poa': 'nzuri', 'freshi': 'nzuri',
            'bora': 'nzuri', 'fiti': 'nzuri', 'safi': 'nzuri', 'moto': 'nzuri',
            'bomba': 'nzuri', 'freshi': 'nzuri',
            # Negative Sheng markers
            'choma': 'mbaya', 'takataka': 'mbaya', 'kubaya': 'mbaya', 
            'kibaya': 'mbaya', 'baya': 'mbaya', 'fala': 'mbaya',
            'pumbavu': 'jinga', 'kijinga': 'jinga', 'ujinga': 'jinga',
            'hawafai': 'mbaya', 'upuuzi': 'mbaya',
            # Intensifiers
            'dah': 'sana', 'kwelikweli': 'sana', 'haki': 'kweli',
            'jameni': 'tafadhali', 'wacha': 'acha', 'maze': 'sana'
        }
        
        # Negations to preserve
        self.negations = {'sio', 'si', 'haku', 'hau', 'ha', 'siwezi', 'sik', 
                         'bila', 'kutokuwa', 'no', 'not', 'never', 'none',
                         'hapana', 'hata kidogo', 'siyo'}
    
    def _get_nltk(self):
        """Lazy load NLTK"""
        if self.nltk is None:
            self.nltk = load_nltk()
        return self.nltk
    
    def clean_text(self, text):
        """Clean text quickly"""
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
        """Normalize Sheng slang using mapping dictionary"""
        words = text.split()
        normalized = [self.sheng_map.get(word, word) for word in words]
        return ' '.join(normalized)
    
    def remove_stopwords(self, text):
        """Remove stopwords efficiently"""
        words = text.split()
        filtered = [w for w in words if w not in self.custom_stopwords or w in self.negations]
        return ' '.join(filtered)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        if not isinstance(text, str):
            return ""
        
        text = self.clean_text(text)
        text = self.handle_sheng_slang(text)
        text = self.remove_stopwords(text)
        
        return text
    
    def batch_preprocess(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_sentiment_pie_chart(sentiment_counts):
    """Create a pie chart for sentiment distribution"""
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        hole=0.4,
        marker_colors=[colors.get(k, '#95a5a6') for k in sentiment_counts.keys()],
        textinfo='percent+label',
        textposition='auto'
    )])
    fig.update_layout(height=450, showlegend=True)
    return fig


def create_sentiment_bar_chart(sentiment_counts):
    """Create a bar chart for sentiment distribution"""
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    fig = go.Figure(data=[go.Bar(
        x=list(sentiment_counts.keys()),
        y=list(sentiment_counts.values()),
        marker_color=[colors.get(k, '#95a5a6') for k in sentiment_counts.keys()],
        text=list(sentiment_counts.values()),
        textposition='auto'
    )])
    fig.update_layout(height=450, xaxis_title="Sentiment", yaxis_title="Count")
    return fig


def create_star_rating_chart(df):
    """Create a bar chart for star rating distribution"""
    if 'Star_Rating' not in df.columns:
        return None
    
    rating_counts = df['Star_Rating'].value_counts().sort_index()
    colors = ['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']
    
    fig = go.Figure(data=[go.Bar(
        x=[f"{r} Star" for r in rating_counts.index],
        y=rating_counts.values,
        marker_color=colors[:len(rating_counts)],
        text=rating_counts.values,
        textposition='auto'
    )])
    fig.update_layout(title="Star Rating Distribution", height=400, xaxis_title="Rating", yaxis_title="Count")
    return fig


def create_code_switching_chart(df):
    """Create a chart for code-switching detection"""
    if 'Code_Switch_Detected' not in df.columns:
        return None
    
    cs_counts = df['Code_Switch_Detected'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=cs_counts.index,
        values=cs_counts.values,
        hole=0.3,
        marker_colors=['#3498db', '#95a5a6'],
        textinfo='percent+label'
    )])
    fig.update_layout(title="Code-Switching Detected", height=400)
    return fig


def create_language_mix_chart(df):
    """Create a chart for language mix distribution"""
    if 'Language_Mix' not in df.columns:
        return None
    
    lang_counts = df['Language_Mix'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=lang_counts.values,
        y=lang_counts.index,
        orientation='h',
        marker_color='#3498db',
        text=lang_counts.values,
        textposition='auto'
    )])
    fig.update_layout(title="Language Mix Distribution", height=400, xaxis_title="Count", yaxis_title="Language Mix")
    return fig


def create_word_cloud(texts, title="Word Cloud"):
    """Create word cloud - simplified"""
    if not texts:
        return None
    
    all_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=600, height=350,
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
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score * 100,
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 50], 'color': '#f8d7da'},
                {'range': [50, 75], 'color': '#fff3cd'},
                {'range': [75, 100], 'color': '#d4edda'}
            ]
        }
    ))
    fig.update_layout(height=280)
    return fig


def create_model_comparison_chart(model_scores):
    """Create model comparison chart"""
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
        yaxis_range=[0, 1]
    )
    return fig


def create_top_words_chart(word_frequencies, title="Most Frequent Words", top_n=12):
    """Create top words chart"""
    top_words = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    fig = go.Figure(data=[go.Bar(
        x=list(top_words.values()),
        y=list(top_words.keys()),
        orientation='h',
        marker_color='#3498db',
        text=list(top_words.values()),
        textposition='outside'
    )])
    fig.update_layout(title=title, height=450, yaxis={'categoryorder': 'total ascending'})
    return fig


def create_sentiment_by_platform_chart(df):
    """Create a stacked bar chart for sentiment by platform"""
    if 'Platform' not in df.columns:
        return None
    
    platform_sentiment = pd.crosstab(df['Platform'], df['Sentiment_Label'], normalize='index') * 100
    
    fig = go.Figure()
    for sentiment in platform_sentiment.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=platform_sentiment.index,
            y=platform_sentiment[sentiment],
            marker_color={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}.get(sentiment, '#95a5a6')
        ))
    
    fig.update_layout(
        title="Sentiment Distribution by Platform (%)",
        barmode='stack',
        height=450,
        xaxis_title="Platform",
        yaxis_title="Percentage (%)"
    )
    return fig


def create_sentiment_by_retailer_chart(df, top_n=8):
    """Create a horizontal bar chart for sentiment by retailer"""
    if 'Retailer' not in df.columns:
        return None
    
    retailer_sentiment = df.groupby('Retailer')['Sentiment_Label'].value_counts(normalize=True).unstack().fillna(0) * 100
    retailer_sentiment = retailer_sentiment.loc[retailer_sentiment.sum(axis=1).nlargest(top_n).index]
    
    fig = go.Figure()
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in retailer_sentiment.columns:
            fig.add_trace(go.Bar(
                name=sentiment,
                y=retailer_sentiment.index,
                x=retailer_sentiment[sentiment],
                orientation='h',
                marker_color={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}.get(sentiment, '#95a5a6')
            ))
    
    fig.update_layout(
        title=f"Sentiment Distribution by Retailer (Top {top_n})",
        barmode='stack',
        height=450,
        xaxis_title="Percentage (%)",
        yaxis_title="Retailer"
    )
    return fig


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Kenyan Retail Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #28a745;
    }
    .sentiment-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #6c757d;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .metric-card h3 {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #2c3e50;
        font-size: 2rem;
        margin: 0;
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
    <h1> Kenyan Retail Sentiment Analysis System</h1>
    <h3>Automated Text Mining for Real-Time Customer Feedback Analysis</h3>
    <p>Supporting English, Swahili, and Sheng code-switching</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", ["Dashboard", "Model Training", "Real-Time Analysis", "Analytics", "Dataset Explorer", "About"])
    
    st.markdown("---")
    if st.session_state.data_loaded:
        st.success(f"Data: {len(st.session_state.df)} records")
    else:
        st.info("No data loaded")
    
    st.markdown("---")
    st.markdown("**Team:**")
    st.markdown("- Timothy Joseph\n- Mukiri Sharon\n- Caleb Ngumbau\n- Osama Mohammed\n- Monicah Gitahi")
    st.markdown("---")
    st.markdown("**Supervisor:** Mr. Vincent Mwai")
    st.markdown("**JKUAT - 2026**")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if page == "Dashboard":
    st.header("Dashboard Overview")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Reviews</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pos_count = len(df[df['Sentiment_Label'] == 'Positive'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Positive</h3>
                <h2 style="color: #27ae60;">{pos_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            neu_count = len(df[df['Sentiment_Label'] == 'Neutral'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Neutral</h3>
                <h2 style="color: #7f8c8d;">{neu_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            neg_count = len(df[df['Sentiment_Label'] == 'Negative'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Negative</h3>
                <h2 style="color: #e74c3c;">{neg_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['Sentiment_Label'].value_counts().to_dict()
            st.plotly_chart(create_sentiment_pie_chart(sentiment_counts), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_sentiment_bar_chart(sentiment_counts), use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            star_chart = create_star_rating_chart(df)
            if star_chart:
                st.plotly_chart(star_chart, use_container_width=True)
        
        with col2:
            cs_chart = create_code_switching_chart(df)
            if cs_chart:
                st.plotly_chart(cs_chart, use_container_width=True)
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            lang_chart = create_language_mix_chart(df)
            if lang_chart:
                st.plotly_chart(lang_chart, use_container_width=True)
        
        with col2:
            platform_chart = create_sentiment_by_platform_chart(df)
            if platform_chart:
                st.plotly_chart(platform_chart, use_container_width=True)
        
        # Sample Reviews
        st.markdown("### Sample Customer Reviews")
        
        filter_option = st.selectbox("Filter by sentiment:", ["All", "Positive", "Neutral", "Negative"])
        
        if filter_option == "Positive":
            sample_df = df[df['Sentiment_Label'] == 'Positive'].head(10)
        elif filter_option == "Negative":
            sample_df = df[df['Sentiment_Label'] == 'Negative'].head(10)
        elif filter_option == "Neutral":
            sample_df = df[df['Sentiment_Label'] == 'Neutral'].head(10)
        else:
            sample_df = df.head(10)
        
        for _, row in sample_df.iterrows():
            sentiment = row['Sentiment_Label']
            if sentiment == 'Positive':
                st.markdown(f"""
                <div class="sentiment-positive">
                    <strong>POSITIVE</strong><br>
                    {row['Review_Text'][:250]}...
                </div>
                """, unsafe_allow_html=True)
            elif sentiment == 'Negative':
                st.markdown(f"""
                <div class="sentiment-negative">
                    <strong>NEGATIVE</strong><br>
                    {row['Review_Text'][:250]}...
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sentiment-neutral">
                    <strong>NEUTRAL</strong><br>
                    {row['Review_Text'][:250]}...
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No data loaded. Please go to Model Training page and upload your dataset.")

# ============================================================================
# MODEL TRAINING PAGE
# ============================================================================

elif page == "Model Training":
    st.header("Model Training")
    
    st.markdown("""
    Train machine learning models (Naive Bayes and SVM) on customer feedback data.
    The system supports code-switching between English, Swahili, and Sheng.
    """)
    
    uploaded_file = st.file_uploader("Upload Excel or CSV file with customer reviews", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, sheet_name='Raw_Dataset')
            else:
                df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'Review_Text' in df.columns and 'Sentiment_Label' in df.columns:
                
                # Keep only relevant columns and drop nulls
                df = df[['Review_Text', 'Sentiment_Label']].dropna()
                
                # Map sentiment labels to numeric (for binary classification, we'll focus on Positive vs Negative)
                # For Neutral, we'll handle separately
                df_binary = df[df['Sentiment_Label'].isin(['Positive', 'Negative'])].copy()
                label_map = {'Positive': 1, 'Negative': 0}
                df_binary['label_encoded'] = df_binary['Sentiment_Label'].map(label_map)
                
                st.session_state.df = df_binary
                st.session_state.data_loaded = True
                
                st.success(f"Successfully loaded {len(df_binary)} reviews (Positive + Negative)!")
                st.write(f"Note: {len(df) - len(df_binary)} neutral reviews excluded from training (kept for analysis only)")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df_binary[['Review_Text', 'Sentiment_Label']].head(10), use_container_width=True)
                
                # Show statistics
                st.subheader("Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total (Binary)", len(df_binary))
                with col2:
                    st.metric("Positive", len(df_binary[df_binary['Sentiment_Label'] == 'Positive']))
                with col3:
                    st.metric("Negative", len(df_binary[df_binary['Sentiment_Label'] == 'Negative']))
            else:
                st.error("Excel file must contain 'Review_Text' and 'Sentiment_Label' columns")
                
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
        
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training model... Please wait."):
                
                if st.session_state.preprocessor is None:
                    st.session_state.preprocessor = TextPreprocessor()
                
                preprocessor = st.session_state.preprocessor
                texts = st.session_state.df['Review_Text'].astype(str).tolist()
                
                with st.spinner("Preprocessing text..."):
                    processed_texts = preprocessor.batch_preprocess(texts)
                
                with st.spinner("Vectorizing text..."):
                    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(processed_texts)
                    y = st.session_state.df['label_encoded'].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                results = {}
                
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
                
                best_model = max(results, key=lambda x: results[x]['accuracy'])
                st.session_state.model = results[best_model]['model']
                st.session_state.vectorizer = vectorizer
                
                joblib.dump(st.session_state.model, 'sentiment_model.pkl')
                joblib.dump(st.session_state.vectorizer, 'vectorizer.pkl')
                
                st.success(f"Training complete! Best model: {best_model}")
                st.metric("Accuracy", f"{results[best_model]['accuracy']:.4f}")
                
                model_scores = {name: {k: v for k, v in m.items() if k != 'model'} 
                               for name, m in results.items()}
                st.plotly_chart(create_model_comparison_chart(model_scores), use_container_width=True)
                
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

elif page == "Real-Time Analysis":
    st.header("Real-Time Sentiment Analysis")
    
    st.markdown("""
    Enter customer feedback below to analyze its sentiment in real-time.
    The system supports English, Swahili, and Sheng (code-switching).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Positive Example", use_container_width=True):
            st.session_state.example_text = "Bidhaa hii ni kali sana, nimefurahishwa na huduma yenu. Delivery ilikuwa bomba!"
    with col2:
        if st.button("Load Negative Example", use_container_width=True):
            st.session_state.example_text = "Walinidanganya. Product quality ni mbaya na nobody is responding to my complaint. Hawafai!"
    
    user_input = st.text_area(
        "Customer Feedback:",
        value=st.session_state.get('example_text', ''),
        height=120,
        placeholder="Example: Bidhaa hii ni nzuri sana, nitaipenda..."
    )
    
    if st.button("Analyze Sentiment", use_container_width=True):
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
                    
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "Positive":
                            st.markdown("""
                            <div class="sentiment-positive" style="padding: 2rem;">
                                <h2 style="margin: 0;">POSITIVE SENTIMENT</h2>
                                <p style="margin-top: 0.5rem;">The customer expressed satisfaction with the product or service.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="sentiment-negative" style="padding: 2rem;">
                                <h2 style="margin: 0;">NEGATIVE SENTIMENT</h2>
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
                st.error("No trained model found. Please go to Model Training page and train a model first.")
        else:
            st.warning("Please enter some text to analyze.")

# ============================================================================
# ANALYTICS PAGE
# ============================================================================

elif page == "Analytics":
    st.header("Advanced Analytics")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        st.subheader("Word Cloud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Reviews")
            pos_texts = df[df['Sentiment_Label'] == 'Positive']['Review_Text'].tolist()
            if pos_texts:
                fig = create_word_cloud(pos_texts, "Positive Reviews Word Cloud")
                if fig:
                    st.pyplot(fig)
            else:
                st.info("No positive reviews found")
        
        with col2:
            st.markdown("#### Negative Reviews")
            neg_texts = df[df['Sentiment_Label'] == 'Negative']['Review_Text'].tolist()
            if neg_texts:
                fig = create_word_cloud(neg_texts, "Negative Reviews Word Cloud")
                if fig:
                    st.pyplot(fig)
            else:
                st.info("No negative reviews found")
        
        st.subheader("Most Frequent Words Analysis")
        
        if st.session_state.preprocessor is None:
            st.session_state.preprocessor = TextPreprocessor()
        
        all_texts = df['Review_Text'].astype(str).tolist()
        processed = st.session_state.preprocessor.batch_preprocess(all_texts)
        all_words = ' '.join(processed).split()
        word_freq = Counter(all_words)
        
        fig_top = create_top_words_chart(word_freq, "Most Frequent Words Across All Reviews")
        st.plotly_chart(fig_top, use_container_width=True)
        
        st.subheader("Sentiment by Retailer")
        retailer_chart = create_sentiment_by_retailer_chart(df)
        if retailer_chart:
            st.plotly_chart(retailer_chart, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Export Analysis Report")
        
        if st.button("Generate Report", use_container_width=True):
            total = len(df)
            pos = len(df[df['Sentiment_Label'] == 'Positive'])
            neu = len(df[df['Sentiment_Label'] == 'Neutral']) if 'Neutral' in df['Sentiment_Label'].values else 0
            neg = len(df[df['Sentiment_Label'] == 'Negative'])
            
            report_data = {
                'metric': ['Total Reviews', 'Positive Reviews', 'Neutral Reviews', 'Negative Reviews', 
                          'Positive Percentage', 'Negative Percentage'],
                'value': [total, pos, neu, neg, f"{(pos/total*100):.1f}%", f"{(neg/total*100):.1f}%"]
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
        st.info("No data loaded. Please go to Model Training page to load and analyze data.")

# ============================================================================
# DATASET EXPLORER PAGE (FIXED)
# ============================================================================

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        
        # Get the full original data (before binary conversion)
        # We need to reload from the uploaded file or store the full dataframe
        if 'full_df' not in st.session_state:
            st.warning("Full dataset not available. Please upload the Excel file again in Model Training page.")
            st.info("Note: The Dataset Explorer requires the original Excel file to show all columns.")
        else:
            df_full = st.session_state.full_df
            
            st.subheader("Full Dataset Overview")
            st.dataframe(df_full.head(20), use_container_width=True)
            
            st.subheader("Dataset Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Retailers by Review Count**")
                retailer_counts = df_full['Retailer'].value_counts().head(10)
                st.bar_chart(retailer_counts)
            
            with col2:
                st.write("**Top Product Categories**")
                category_counts = df_full['Product_Category'].value_counts().head(10)
                st.bar_chart(category_counts)
            
            st.subheader("Filter Reviews")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                retailer_options = ["All"] + sorted(df_full['Retailer'].unique().tolist())
                retailer_filter = st.selectbox("Select Retailer", retailer_options)
            with col2:
                sentiment_options = ["All"] + sorted(df_full['Sentiment_Label'].unique().tolist())
                sentiment_filter = st.selectbox("Select Sentiment", sentiment_options)
            with col3:
                lang_options = ["All"] + sorted(df_full['Language_Mix'].unique().tolist())
                lang_filter = st.selectbox("Select Language Mix", lang_options)
            
            filtered_df = df_full.copy()
            if retailer_filter != "All":
                filtered_df = filtered_df[filtered_df['Retailer'] == retailer_filter]
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df['Sentiment_Label'] == sentiment_filter]
            if lang_filter != "All":
                filtered_df = filtered_df[filtered_df['Language_Mix'] == lang_filter]
            
            st.write(f"Showing {len(filtered_df)} reviews")
            
            # Select columns to display
            display_cols = ['Review_ID', 'Review_Date', 'Retailer', 'Product_Category', 
                           'Star_Rating', 'Sentiment_Label', 'Language_Mix', 'Code_Switch_Detected', 'Review_Text']
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[available_cols].head(20), use_container_width=True)
    else:
        st.info("No data loaded. Please go to Model Training page and upload your Excel file first.")

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
    
    ### Dataset Information
    
    The **Kenyan Retail Sentiment Dataset** contains 2,000 customer reviews collected from major
    Kenyan e-commerce platforms including:
    
    - Jumia
    - Kilimall
    - Masoko
    - Copia
    - Sky.Garden
    - Naivas Digital
    - Pigiame
    - Tuskys Online
    - Flare
    - Jiji
    
    Each review includes:
    - Review text (English, Swahili, Sheng)
    - Sentiment label (Positive, Neutral, Negative)
    - Star rating (1-5)
    - Platform and retailer information
    - Language mix and code-switching detection
    
    ---
    
    ### Methodology (DSR - Design Science Research)
    
    The project follows the Design Science Research methodology:
    
    - **Problem Identification** - Manual feedback analysis; code-switching challenges
    - **Define Objectives** - Three specific objectives (preprocessing, training, deployment)
    - **Design & Development** - Build preprocessing pipeline, train models, develop web app
    - **Demonstration** - Run the application with the Kenyan dataset
    - **Evaluation** - Accuracy, precision, recall, F1-score
    - **Communication** - This document and application deployment
    
    ---
    
    ### Technologies Used
    
    | Technology | Purpose |
    |------------|---------|
    | Python 3.10+ | Core programming language |
    | Streamlit | Web application framework |
    | Scikit-learn | Machine learning algorithms (Naive Bayes, SVM) |
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
        
        df = st.session_state.df
        total = len(df)
        pos = len(df[df['Sentiment_Label'] == 'Positive'])
        neg = len(df[df['Sentiment_Label'] == 'Negative'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records (Binary)", total)
        with col2:
            st.metric("Positive Reviews", pos)
        with col3:
            st.metric("Negative Reviews", neg)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; padding: 1rem;'>"
    "(c) 2026 - Kenyan Retail Sentiment Analysis System | JKUAT"
    "</div>",
    unsafe_allow_html=True
)
