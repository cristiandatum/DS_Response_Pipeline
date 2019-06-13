import sys

import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger']) # download for lemmatization

import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import pickle
import datetime

from sqlalchemy import create_engine

def load_data(database_filepath):

    '''
    loads the database with messages

    Input:
    database_filepath: from user

    Output:
    - X: tweeter message
    - Y: classification eg. aid_centers, aid_related, buildings, etc.
    - df: dataframe
    '''

    engine = create_engine('sqlite:///DisasterResponseDatabase.db')

    df = pd.read_sql_table('MessageClassification',con='sqlite:///DisasterResponseDatabase.db')
    X = df.iloc[:,1] 
    Y = df.iloc[:,4:] 

    return df,X,Y

def tokenize(text):
    '''
    converts tweet messages into simplified lemmitized words.

    Input: 
    text: input single message as a string

    Output:
    words: the sentence converted into separate words list
    '''

    text = text.lower() #makes text lower case
    text = re.sub(r"[^a-zA-Z0-9]"," ",text) #removes non-alphabetic characters
    words=word_tokenize(text) #splits into words

    words = [w for w in words if w not in stopwords.words("english")]# Remove stop words    
    words = [PorterStemmer().stem(w) for w in words] #apply stemming to words (eg. branching -> branch)
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]#apply lemmatizing to words (eg. was -> is)
    
    return words


def build_model():
    '''
    Create a LinearSVC model using MultiOutputClassifier.

    The parameters and model were selected based on GridSearchCV iterations.

    Output:
    returns LinearSVC model.
    '''

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)), #create the CountVectorizer object
        ('tfidf', TfidfTransformer()), #create Tfidftransformer object    
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC()))) #create the Classifier object
    ])

    #parameters identified from GridCV search
    parameters = {   
        'clf__estimator__estimator__C': 1,
        'tfidf__use_idf': False,
        'vectorizer__max_df': 0.8,
        'vectorizer__ngram_range': (1, 1)
    }

    #create a grid searchCV for clarity of code
    grid_cv = GridSearchCV(pipeline, param_grid=parameters, cv=5,verbose=3,n_jobs=-1)

    return grid_cv

def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Prints out statistical analysis (precision, recall and accuracy).

    Input:
    - model: NLP model
    - X_test: 20% of messages in dataset (validation) 
    - Y_test: 20% of classifications in dataset (validation)
    - category_names: classification titles

    Output:
    prints the statistical analysis
    '''

    Y_pred = model.predict(X_test)

    Y_test=pd.DataFrame(data=Y_test,columns=Y.columns)     #Convert prediction numpy into dataframe
    Y_pred=pd.DataFrame(data=Y_pred,columns=Y.columns)
    
    for column in Y_pred.columns:
        print(column)
        print(classification_report(Y_test[column], Y_pred[column]))
        print('_____________________________________________________')


def save_model(model, model_filepath):
    pass

    pickle_out = open('data/model.pkl','wb')
    pickle.dump(model, pickle_out)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()