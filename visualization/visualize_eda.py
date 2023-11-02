import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Function to create exploratory visualizations
def create_exploratory_visualizations(df, exploratory_folder):
    # Box plot for 'similarity', 'length_diff', 'ref_tox', 'trn_tox'
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    data = [df['similarity'], df['length_diff'], df['ref_tox'], df['trn_tox']]
    labels = ['Similarity', 'Length Difference', 'Reference Toxicity', 'Translation Toxicity']
    for i in range(2):
        for j in range(2):
            axs[i, j].boxplot(data[i * 2 + j])
            axs[i, j].set_ylabel('Values')
            axs[i, j].set_title(labels[i * 2 + j])
    plt.tight_layout()
    plt.savefig(os.path.join(exploratory_folder, 'box_plots.png'))
    plt.close()

    # Histograms for 'similarity', 'length_diff', 'ref_tox', 'trn_tox'
    plt.figure(figsize=(6, 4))
    sns.histplot(df['similarity'], bins=25, kde=True)
    plt.title('Distribution of Similarity')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(exploratory_folder, 'similarity_distribution.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['length_diff'], bins=30, kde=True)
    plt.title('Distribution of Length Difference')
    plt.xlabel('Length Difference')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(exploratory_folder, 'length_diff_distribution.png'))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df['ref_tox'], bins=30, kde=True)
    plt.title('Distribution of Reference Toxicity')
    plt.xlabel('Reference Toxicity')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(df['trn_tox'], bins=30, kde=True)
    plt.title('Distribution of Translation Toxicity')
    plt.xlabel('Translation Toxicity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(exploratory_folder, 'toxicity_distributions.png'))
    plt.close()

    # Correlation matrix heatmap
    correlation_matrix = df[['similarity', 'length_diff', 'ref_tox', 'trn_tox']].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(exploratory_folder, 'correlation_matrix.png'))
    plt.close()

    # Scatter plot for 'ref_tox' vs. 'trn_tox'
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='ref_tox', y='trn_tox', data=df)
    plt.title('Translation Toxicity vs. Reference Toxicity')
    plt.xlabel('Reference Toxicity')
    plt.ylabel('Translation Toxicity')
    plt.savefig(os.path.join(exploratory_folder, 'toxicity_scatter.png'))
    plt.close()

    # Text Length Distributions
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df['reference_length'], bins=30, kde=True)
    plt.title('Distribution of Reference Text Length')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(df['translation_length'], bins=30, kde=True)
    plt.title('Distribution of Translation Text Length')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(exploratory_folder, 'text_length_distributions.png'))
    plt.close()

    # Word Clouds
    reference_text = ' '.join(df['reference'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reference_text)
    plt.figure(figsize=(7, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Reference Text')
    plt.savefig(os.path.join(exploratory_folder, 'reference_wordcloud.png'))
    plt.close()

    translation_text = ' '.join(df['translation'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(translation_text)
    plt.figure(figsize=(7, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Translation Text')
    plt.savefig(os.path.join(exploratory_folder, 'translation_wordcloud.png'))
    plt.close()

    # Distribution of Stop Words
    stop_words = set(stopwords.words('english'))
    df['reference_stopwords'] = df['reference'].apply(lambda x: len([word for word in str(x).lower().split() if word in stop_words]))
    df['translation_stopwords'] = df['translation'].apply(lambda x: len([word for word in str(x).lower().split() if word in stop_words]))

    plt.figure(figsize=(8, 4))
    sns.histplot(df['reference_stopwords'], bins=15, kde=True, label='Reference')
    sns.histplot(df['translation_stopwords'], bins=15, kde=True, label='Translation')
    plt.title('Distribution of Stop Words')
    plt.xlabel('Number of Stop Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(exploratory_folder, 'stopwords_distribution.png'))
    plt.close()
    
    # Top 20 Most Frequent Unigrams
    vectorizer = CountVectorizer(stop_words='english')
    unigrams = vectorizer.fit_transform(df['text_combined'])
    unigram_frequencies = np.array(unigrams.sum(axis=0)).squeeze()
    feature_names = vectorizer.get_feature_names_out()

    unigram_freq_df = pd.DataFrame({'unigram': feature_names, 'frequency': unigram_frequencies})
    unigram_freq_df = unigram_freq_df.sort_values(by='frequency', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='frequency', y='unigram', data=unigram_freq_df.head(20))
    plt.xlabel('Frequency')
    plt.ylabel('Unigram')
    plt.title('Top 20 Most Frequent Unigrams')
    plt.savefig(os.path.join(exploratory_folder, 'top_unigrams.png'))
    plt.close()

    # Distribution of Unigram Lengths
    vectorizer = CountVectorizer(stop_words='english')
    unigrams = vectorizer.fit_transform(df['text_combined'])
    unigram_lengths = np.array(unigrams.sum(axis=1)).squeeze()

    plt.figure(figsize=(6, 4))
    sns.histplot(unigram_lengths, bins=50, kde=True)
    plt.xlabel('Unigram Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Unigram Lengths')
    plt.savefig(os.path.join(exploratory_folder, 'unigram_length_distribution.png'))
    plt.close()

# Main function to generate visualizations
def generate_visualizations(df):
    exploratory_folder = '../reports/figures/exploratory'
    os.makedirs(exploratory_folder, exist_ok=True)
    create_exploratory_visualizations(df, exploratory_folder)
