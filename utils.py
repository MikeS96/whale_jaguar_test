import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Most frequent Unigrams
# https://www.kaggle.com/imdevskp/news-category-classification
def category_ngram(category, n, df):
    # Create Dataframe of subgroup element
    temp_df = df[df['category'] == category]
    # Convert text into a matrix of token counts (object) --> Sparse matrix
    word_vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')
    # Fit the model to the dataframe
    sparse_matrix = word_vectorizer.fit_transform(temp_df['headline'])
    # Count the frequency
    frequencies = sum(sparse_matrix).toarray()[0]
    # Show the results as a dataframe
    return pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])\
            .sort_values(by='frequency', ascending=False) \
            .reset_index() \
            .head(10)

# Plot history (accuracy and loss)
# https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
def plot_history(history):
    
    plt.figure(figsize=(20, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Training Accuracy', c = 'dodgerblue', lw = '2')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy', c = 'orange', lw = '2')
    plt.title('Accuracy', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Training Loss', c = 'dodgerblue', lw = '2')
    plt.plot(history.history['val_loss'], label = 'Validation Loss', c = 'orange', lw = '2')
    plt.title('Loss', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot confusion matrix
def plot_cmatrix(pred, ticklabels, figsize, val_labels):
      
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

    cm = confusion_matrix(val_labels, pred)
    sns.heatmap(cm, annot=True, cbar=False, fmt='1d', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_yticklabels(ticklabels, rotation=0)

    plt.show()
