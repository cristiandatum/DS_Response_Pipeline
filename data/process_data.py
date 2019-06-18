import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads files with Tweeter messages and categories and returns dataframe
    
    Input: 
    message_filepath: CSV file with messages
    categories_filepath: CSV file with categoriess

    Output:
    - df: merged Dataframe
    '''

    messages = pd.read_csv(messages_filepath) #load csv file into pd
    categories = pd.read_csv(categories_filepath) #load csv files into pd
    df = pd.merge(messages, categories, on='id') #merge dataframes
    return df

def clean_data(df):
    '''
    Cleans the dataframe
    
    Input:
    - df: Dataframe from load_data

    Output:
    - df: clean dataframe


    '''
    categories =  df['categories'].str.split(";",expand=True)
  
    #select the first row of the categories dataframe to extract column names 
    category_colnames = categories.iloc[0].str.slice(stop=-2)

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    #arrange columns in alphabetical order
    cols = ['id','message','original','genre','aid_centers', \
        'aid_related', 'buildings', 'child_alone', 'clothing',\
        'cold', 'death', 'direct_report', 'earthquake', 'electricity',\
        'fire','floods', 'food',  'hospitals',  'infrastructure_related',\
        'medical_help', 'medical_products',  'military','missing_people',\
        'money', 'offer',  'other_aid','other_infrastructure', \
        'other_weather', 'refugees', 'related','request', \
        'search_and_rescue', 'security', 'shelter', 'shops', 'storm',\
        'tools', 'transport', 'water', 'weather_related']
    df = df[cols]

    #drop duplicate columns
    df.drop_duplicates(subset ="id", keep = False, inplace = True)
    
    return df

def save_data(df, database_filename):
    '''
    Saves dataframe into a database
    
    Input:
    - df: Clean Dataframe
    - database_filename: filename from user

    Ouput:
    Database saved in filepath name
    '''
    #create engine: engine
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageClassification', engine, index=False, if_exists='replace')

  
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()