import re
import pandas as pd
import spacy
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('sample_data.csv')

# Loading the German language model 'de_core_news_sm' from spaCy, to use NLP capabilities for the German language
nlp = spacy.load('de_core_news_sm')

# Removing rows with missing labels
data.dropna(inplace=True)

# Removing rows with empty or space-only text entries
space_or_empty_text=data[(data['text'].apply(lambda x: len(x.split())) == 0)]
data_cleaned = data.drop(space_or_empty_text.index)

# Defining a function for text preprocessing
def preprocess_text(text):
    """
    Preprocesses text data by performing the following steps:
    1. Converts text to lowercase for consistency.
    2. Removes all characters except letters (a-z), German umlauts (ä, ö, ü), and spaces.
    3. Tokenizes and lemmatizes the text using spaCy, removing stop words and punctuation.
    
    Args:
        text (str): The input text to be preprocessed.
        
    Returns:
        str: The preprocessed text with removed noise, lemmatized, and joined as a string.
    """
    text = text.lower()  
    text = re.sub(r'[^a-zäöüß\s]', '', text) 
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

#Applying the 'preprocess_text' function to each element in the 'text' column of the DataFrame
data['processed_text'] = data['text'].apply(preprocess_text)

#Identifying unique labels in the 'label' column of the dataset
unique_labels = data['label'].unique()

#Initilaizing a LabelEncoder to transform these labels into numerical values
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)

data['encoded_label'] = label_encoder.fit_transform(data['label'])

#Splitting the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['encoded_label'], test_size=0.2, random_state=42)

#Utilizing TF-IDF vectorization to convert text data in the training and testing sets (X_train and X_test) into numerical feature vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing and training a classifier model using the TF-IDF transformed training data  
# and corresponding labels 
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
model.fit(X_train_vec, y_train)

# Saving the trained model
dump(model, 'logistic_regression_model.joblib')
dump(vectorizer, 'tfidf_vectorizer.joblib')
dump(label_encoder, 'label_encoder.joblib')

