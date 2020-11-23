import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Most frequent Unigrams
# https://www.kaggle.com/imdevskp/news-category-classification
def category_ngram(category, n, df, feature, sub_category):
    # Create Dataframe of subgroup element
    temp_df = df[df[feature] == category]
    # Convert text into a matrix of token counts (object) --> Sparse matrix
    word_vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')
    # Fit the model to the dataframe
    sparse_matrix = word_vectorizer.fit_transform(temp_df[sub_category])
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

# Plot wordCloud
def display_wordcloud(authors_df, stop_words, n_components, feature2filter, varText):
    plt.figure(figsize = (16, 8), facecolor = None)
    j = np.ceil(n_components / 4)
    for t in range(n_components):
        current_author = authors_df[feature2filter].unique()[t]
        cat_df = category_ngram(current_author, 1, 
                                authors_df, feature2filter, varText)
        cat_tuple = [tuple(x) for x in cat_df.values]
        i=t+1
        plt.subplot(j, 4, i).set_title(current_author.split(',')[0])
        plt.plot()
        wordcloud = WordCloud(max_words = 200, background_color ='white', 
                              stopwords = stop_words).generate_from_frequencies(dict(cat_tuple))
        plt.imshow(wordcloud)
        plt.axis("off")
    plt.show()

# Create dictionary of authors and frecuency per word
def aut_frequency(iterations, authors_name_dic, full_authors_df, max_words):
    authors_dic = {'authors': []}
    for i in range(iterations):
        # Obtain the author in the current index
        author_name = authors_name_dic['authors'][i]
        # Obtain a DF with the most frequent words of an author (Take only two most common words)
        aut_df = category_ngram(author_name, 1, full_authors_df, 'authors', 'full_text')[:max_words]
        # Create a dictionary out of the last df
        current_aut_dict = dict([tuple(x) for x in aut_df.values])
        # Append the new author
        authors_dic['authors'].append(author_name)
        # Fill the dictionary
        for aut in current_aut_dict:
            # If the key already exists, append the frequency for the given author
            if aut in authors_dic.keys():
                # Append the number of occurrences of a word for this author
                authors_dic[aut].append(current_aut_dict[aut])
            # otherwise create the key with its value
            else: 
                # Create a new list for this item
                authors_dic[aut] = list(np.zeros(i))
                authors_dic[aut].append(current_aut_dict[aut])
        # For loop to fill the unfilled values with zeros for a given author
        expected_size = i + 1
        #print(authors_dic.keys())
        for headers in authors_dic.keys():
            # Lenght of a given key
            len_header = len(authors_dic[headers])
            if expected_size != len_header:
                # Append zeros
                diff_size = expected_size - len_header
                for j in range(diff_size):
                    authors_dic[headers].append(0)

    return pd.DataFrame.from_dict(authors_dic)