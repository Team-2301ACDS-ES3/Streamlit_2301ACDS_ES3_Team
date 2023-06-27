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
	new_text = text.translate(str.maketrans('','',string.punctuation))
	
	return new_text

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
	st.title("COMBIS Classifier")
	st.subheader("A Climate change tweet classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Information", "Prediction", "About us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		st.image('resources/imgs/climate_01.jpg', width = 700)
		st.subheader("Welcome to the Tweet Classifier. The Climate change tweet classification interactive app")
		st.subheader("brought to you by ``COMBIS Analytics``:registered:")
		info_markdown = read_markdown_file("resources/home.md")
		st.markdown(info_markdown)
		st.write("Hint: Explore through the side bar")

	# Building out the "Information" page
	if selection == "Information":
		options = ["General Information", "Exploratory Data Analysis", "Model Information"]
		selection = st.sidebar.selectbox("Choose Option", options)

		if selection == "General Information":
			st.info("General Information")
			# You can read a markdown file from supporting resources folder
			info_markdown = read_markdown_file("resources/info.md")
			st.markdown(info_markdown)

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page
		
		if selection == "Exploratory Data Analysis":
			st.info("Exploratory Data Analysis")
			# Building out the EDA page
			st.subheader("Visual insight from the given data")
			st.image('resources/imgs/EDA_01.png', width = 700)
			st.write("Over 8,000 tweets were 'Pro', compared to the slightly above 1,000 'Anti' sentiments")
			st.subheader("")# creating space between the texts and images
			st.subheader("")

			st.image('resources/imgs/EDA_02.png', width = 700)
			st.write("Here the Pie chart shows that more than half of the tweet(53.9%) support the belief that climate change is man made")
			st.subheader("")# creating space between the texts and images
			st.subheader("")

			st.image('resources/imgs/EDA_03.png', width = 700)
			st.write("The wordcloud image shows the most occurring words in the data set")
			st.write("Climate change and Global warming are the most used with different beliefs")
			st.subheader("")# creating space between the texts and images
			st.subheader("")

			st.image('resources/imgs/EDA_04.png', width = 800)
			st.write("The wordcloud image here shows the most occurring words in each sentiment class")
			st.write("The visuals demonstrates the accuracy of the data source")
			st.subheader("")# creating space between the texts and images
			st.subheader("")

		if selection == "Model Information":
			st.info("Model Information")
			# You can read a markdown file from supporting resources folder
			info_markdown = read_markdown_file("resources/models.md")
			st.markdown(info_markdown)

	# Building out the predication page
	if selection == "Prediction":
		options = ["Logistic Regression", "Support Vector Machines", "Naive Bayes", "Random Forest", "K-Nearest Neighbours"]
		selection = st.sidebar.selectbox("Choose a Model", options)

		if selection == "Logistic Regression":
			st.info("Prediction with Logistic Regression")
			st.write("Hello Earth! We will heal: ``Text Categorized as: [0] ↔️ Neutral😐``")
			st.write("Climate change is man made: ``Text Categorized as: [1] ↔️ Pro✅``")
			st.write("It is a wrong narrative to suggest that global warming is man made: ``Text Categorized as: [-1] ↔️ Anti❌``")
			st.write("Weather report for Miami this morning suggests a rainy day: ``Text Categorized as: [2] ↔️ News📰``")
			st.subheader("")
			st.subheader("")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_clean(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:newspaper:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:white_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:neutral_face:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))

		if selection == "Support Vector Machines":
			st.info("Prediction with Support Vector Machines")
			st.info("Prediction with Logistic Regression")
			st.write("Hello Earth! We will heal: ``Text Categorized as: [0] ↔️ Neutral😐``")
			st.write("Climate change is man made: ``Text Categorized as: [1] ↔️ Pro✅``")
			st.write("It is a wrong narrative to suggest that global warming is man made: ``Text Categorized as: [-1] ↔️ Anti❌``")
			st.write("Weather report for Miami this morning suggests a rainy day: ``Text Categorized as: [2] ↔️ News📰``")
			st.subheader("")
			st.subheader("")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_clean(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/SVM.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:newspaper:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:white_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:neutral_face:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))
		
		if selection == "Naive Bayes":
			st.info("Prediction with Naive Bayes")
			st.info("Prediction with Logistic Regression")
			st.write("Hello Earth! We will heal: ``Text Categorized as: [0] ↔️ Neutral😐``")
			st.write("Climate change is man made: ``Text Categorized as: [1] ↔️ Pro✅``")
			st.write("It is a wrong narrative to suggest that global warming is man made: ``Text Categorized as: [-1] ↔️ Anti❌``")
			st.write("Weather report for Miami this morning suggests a rainy day: ``Text Categorized as: [2] ↔️ News📰``")
			st.subheader("")
			st.subheader("")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_clean(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/naive_bayes.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:newspaper:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:white_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:neutral_face:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))

		if selection == "Random Forest":
			st.info("Prediction with Random Forest")
			st.info("Prediction with Logistic Regression")
			st.write("Hello Earth! We will heal: ``Text Categorized as: [0] ↔️ Neutral😐``")
			st.write("Climate change is man made: ``Text Categorized as: [1] ↔️ Pro✅``")
			st.write("It is a wrong narrative to suggest that global warming is man made: ``Text Categorized as: [-1] ↔️ Anti❌``")
			st.write("Weather report for Miami this morning suggests a rainy day: ``Text Categorized as: [2] ↔️ News📰``")
			st.subheader("")
			st.subheader("")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_clean(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/forest.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:newspaper:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:white_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:neutral_face:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))

		if selection == "K-Nearest Neighbours":
			st.info("Prediction with K-Nearest Neighbours")
			st.info("Prediction with Logistic Regression")
			st.write("Hello Earth! We will heal: ``Text Categorized as: [0] ↔️ Neutral😐``")
			st.write("Climate change is man made: ``Text Categorized as: [1] ↔️ Pro✅``")
			st.write("It is a wrong narrative to suggest that global warming is man made: ``Text Categorized as: [-1] ↔️ Anti❌``")
			st.write("Weather report for Miami this morning suggests a rainy day: ``Text Categorized as: [2] ↔️ News📰``")
			st.subheader("")
			st.subheader("")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_clean(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/knn.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:newspaper:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:white_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:neutral_face:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))

	# Building out the 'About us' page
	if selection == "About us":
			st.info("About Us")
			# Building out the 'about us' page
			st.image('resources/imgs/tech_team_01.jpg', width= 700)
			st.subheader("`Mission Statement:`:page_with_curl::page_with_curl:")
			st.subheader("At COMBIS Tech:registered:, we specialize in transforming challenges into\
						growth opportunities. With a focus on innovation and expertise, we help\
						companies thrive in today's dynamic business landscape. Harnessing the\
						power of advanced technology and data-driven insights, we provide tailored\
						solutions that drive remarkable results.")
			st.subheader("We pride ourselves in applying the latest technology to provide actionable intel,\
						in turn helping businesses to grow and nurture a consumer-first mindset.")
			st.subheader("   ")# just a way to create space between texts
			st.subheader("   ")
			st.subheader("`Meet The Team:`:male-technologist::female-factory-worker:")

			st.subheader("   ")
			# First Member
			st.image('resources/imgs/Mati.jpeg', width = 300)
			st.subheader('Mmatlou Matlakala')
			st.write('`Data Scientist:`', '`Team Lead`')
			
			st.subheader("   ")
			# Second Member
			st.image('resources/imgs/Buchi.jpeg', width = 300)
			st.subheader('Onyebuchi Madubuko')
			st.write('`Data Scientist:`', '`Tech Lead`')

			st.subheader("   ")
			# Third Member
			st.image('resources/imgs/Sumaya.jpeg', width = 300)
			st.subheader('Sumaya Hassan')
			st.write('`Data Scientist:`', '`Admin Lead`')

			st.subheader("   ")
			# Fourth Member
			st.image('resources/imgs/Cinta.jpeg', width = 300)
			st.subheader('Jacinta Muindi')
			st.write('`Data Scientist:`', '`Product Lead`')

			st.subheader("   ")
			# Fifth Member
			st.image('resources/imgs/Bolanle.jpeg', width = 300)
			st.subheader('Ipaye Bolanle Moshood')
			st.write('`Data Scientist:`', '`PR Lead`')

			st.subheader("   ")
			# Sixth Member
			st.image('resources/imgs/Omolaja.jpeg', width = 300)
			st.subheader('Omalaja Adodo')
			st.write('`Data Scientist`')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
