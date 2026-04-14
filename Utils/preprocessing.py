"""
Text preprocessing utilities for Swahili/English code-switched text
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    """
    Custom text preprocessor for Swahili/English/Sheng text
    """
    
    def __init__(self):
        # Custom stopwords combining English and Swahili
        self.custom_stopwords = set(stopwords.words('english')).union({
            # Swahili stopwords
            'na', 'ya', 'wa', 'ni', 'cha', 'vya', 'kwa', 'kwenye', 'katika',
            'kuwa', 'kuna', 'kutoka', 'kama', 'hii', 'hizi', 'hicho', 'ile',
            'hao', 'hawa', 'wale', 'yule', 'zile', 'kwamba', 'ingawa', 'basi',
            'ndio', 'ndiyo', 'hivyo', 'hapo', 'huko', 'huku', 'pale', 'kule',
            'nyuma', 'mbele', 'juu', 'chini', 'ndani', 'nje', 'baada', 'kabla',
            'wakati', 'muda', 'sasa', 'bado', 'tena', 'zaidi', 'pia', 'sana',
            'kidogo', 'tu', 'hata', 'mara', 'za', 'la', 'ali', 'ana', 'nil',
            'tuli', 'mli', 'wali', 'nina', 'una', 'ana', 'tuna', 'mna', 'wana',
            'nime', 'ume', 'ame', 'tume', 'mme', 'wame', 'si', 'hu', 'ki', 'vi'
        })
    
    def clean_text(self, text):
        """
        Clean text by removing noise elements
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep the word part)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters and digits (but keep Swahili letters)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_sheng_slang(self, text):
        """
        Normalize common Sheng slang patterns
        """
        # Common Sheng mappings
        sheng_map = {
            'bie': 'nzuri',
            'ngori': 'ngumu',
            'kali': 'nzuri',
            'poa': 'nzuri',
            'freshi': 'nzuri',
            'bora': 'nzuri',
            'choma': 'mbaya',
            'takataka': 'mbaya',
            'kubaya': 'mbaya',
            'kibaya': 'mbaya',
            'baya': 'mbaya',
            'pumbavu': 'jinga',
            'kijinga': 'jinga',
            'ujinga': 'jinga',
            'fiti': 'nzuri',
            'safi': 'nzuri',
            'moto': 'nzuri',
            'dah': 'sana',
            'kwelikweli': 'sana',
            'haki': 'kweli',
            'jameni': 'tafadhali',
            'wacha': 'acha'
        }
        
        words = text.split()
        normalized_words = [sheng_map.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text):
        """
        Remove stopwords while preserving negation words
        """
        words = word_tokenize(text)
        
        # Negation words to preserve (critical for sentiment)
        negations = {'sio', 'si', 'haku', 'hau', 'ha', 'siwezi', 'sik', 
                    'bila', 'kutokuwa', 'no', 'not', 'never', 'none'}
        
        # Also preserve intensifiers
        intensifiers = {'sana', 'kidogo', 'zaidi', 'very', 'too', 'so', 'really'}
        
        filtered_words = [
            word for word in words 
            if word not in self.custom_stopwords 
            or word in negations 
            or word in intensifiers
        ]
        
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        text = self.clean_text(text)
        text = self.handle_sheng_slang(text)
        text = self.remove_stopwords(text)
        
        return text
    
    def batch_preprocess(self, texts):
        """
        Preprocess a batch of texts
        """
        return [self.preprocess(text) for text in texts]


def load_and_prepare_data(filepath):
    """
    Load dataset and prepare for modeling
    """
    df = pd.read_csv(filepath)
    
    # Remove any unnamed index columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    # Ensure labels are consistent
    if 'labels' in df.columns:
        # Map labels to binary if needed (positive=1, negative=0)
        label_map = {'positive': 1, 'negative': 0}
        if df['labels'].dtype == 'object':
            df['label_encoded'] = df['labels'].map(label_map)
        else:
            df['label_encoded'] = df['labels']
    
    return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "team 2019merimera alikuwa takataka",
        "yeye ni mrembo sana",
        "sijafurahishwa na bidhaa hii",
        "mtandao wenu ni mbaya sana siku hizi"
    ]
    
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print("-" * 50)