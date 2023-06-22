"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import string
from pathlib import Path

# Importing markdowns
def read_markdown_file(markdown_file):
	return Path(markdown_file).read_text()

# Text cleaning
def data_clean(text):
	# convert to lower case
	text = text.lower()
	# remove '@' and '#' signs
	text = text.replace('@','').replace('#','')
	# remove punctuations
	clean_text = text.translate(str.maketrans('','',string.punctuation))
	
	return clean_text

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("A Climate change tweet classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Information", "Prediction", "About us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		st.image('resources/imgs/climate_01.jpg', width = 700)
		st.subheader("Welcome to the Tweet Classifier. The Climate change tweet classification interactive app")
		st.subheader("brought to you by ``COMBIS Tech``")
		st.write("Hint: Explore through the side bar")

	# Building out the "Information" page
	if selection == "Information":
		options = ["General Information", "Exploratory Data Analysis", "Model Information"]
		selection = st.sidebar.selectbox("Choose Option", options)

		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
