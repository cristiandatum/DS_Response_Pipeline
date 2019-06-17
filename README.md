# DS_Response_Pipeline
### Disaster Reponse Pipeline for Udacity Data Science nanodegree

### Installation: 


Clone the GitHub repository and use Anaconda distribution of Python 3.
<code:
$ git clone https://..... .git
The code can be viewed and modified with Jupyter Notebooks.

This will require pip install:

sqlalchemy
pickle
nltk


### Instructions:

    Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    Run the following command in the app's directory to run your web app. python run.py

    Go to http://0.0.0.0:3001/

