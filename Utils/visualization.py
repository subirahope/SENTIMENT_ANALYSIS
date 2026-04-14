"""
Visualization utilities for the sentiment analysis dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_sentiment_pie_chart(sentiment_counts):
    """
    Create a pie chart for sentiment distribution using Plotly
    """
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
        width=500,
        showlegend=True
    )
    
    return fig


def create_sentiment_bar_chart(sentiment_counts):
    """
    Create a bar chart for sentiment distribution
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(sentiment_counts.keys()),
            y=list(sentiment_counts.values()),
            marker_color=['#2ecc71', '#e74c3c', '#95a5a6'],
            text=list(sentiment_counts.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution (Bar Chart)",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=450,
        width=500
    )
    
    return fig


def create_trend_line_chart(df, date_column='date'):
    """
    Create a line chart showing sentiment trends over time
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        daily_sentiment = df.groupby([df[date_column].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        
        for sentiment in daily_sentiment.columns:
            fig.add_trace(go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment[sentiment],
                name=sentiment.capitalize(),
                line=dict(color=colors.get(sentiment, '#95a5a6'), width=2),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    return None


def create_word_cloud(texts, title="Word Cloud"):
    """
    Create a word cloud from text data
    """
    if not texts:
        return None
    
    all_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    
    return fig


def create_confidence_gauge(confidence_score):
    """
    Create a gauge chart for prediction confidence
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score * 100,
        title={'text': "Prediction Confidence"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def create_confusion_matrix_plot(confusion_matrix, labels=['Positive', 'Negative']):
    """
    Create a confusion matrix heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=450,
        width=500
    )
    
    return fig


def create_model_comparison_chart(model_scores):
    """
    Create a comparison chart for multiple models
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for model_name, scores in model_scores.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=[scores.get(m.lower(), 0) for m in metrics],
            text=[f'{scores.get(m.lower(), 0):.3f}' for m in metrics],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode='group',
        height=500,
        yaxis_range=[0, 1]
    )
    
    return fig


def create_top_words_chart(word_frequencies, title="Most Frequent Words", top_n=15):
    """
    Create a horizontal bar chart for top words
    """
    top_words = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            orientation='h',
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency",
        yaxis_title="Word",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig