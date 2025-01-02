import os
import ssl
import nltk
import pandas as pd
import streamlit as st
import random
import csv
import datetime
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL & NLTK Setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Check if vectorizer and model exist, otherwise train and save them
if os.path.exists('vectorizer.pkl') and os.path.exists('classifier.pkl'):
    vectorizer = joblib.load('vectorizer.pkl')
    clf = joblib.load('classifier.pkl')
else:
    # Create the vectorizer and classifier
    vectorizer = TfidfVectorizer()
    clf = LogisticRegression(random_state=0, max_iter=10000)

    # Preprocess the data
    tags = []
    patterns = []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)

    # Training the model
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)

    # Save the trained model and vectorizer
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(clf, 'classifier.pkl')

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that. Can you rephrase?"

# Title of the app
st.title('🤖 AI Chatbot')
st.write("Ask me anything and I'll do my best to help!")

# Initialize chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create a sidebar menu with options
menu = ['Home', 'Conversation History', 'About']
choice = st.sidebar.selectbox('Menu', menu)

# Home Menu
if choice == 'Home':
    st.write('Welcome to chatbot. Please type a message and press enter to start')

    # Check if the chat log file exists, and if not, create it with column names
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    # Text input with callback to reset the field
    def on_user_input_change():
        # This will be triggered when user presses Enter or sends the message
        if st.session_state['user_input']:
            # Append user input to chat history
            st.session_state['chat_history'].append(('You', st.session_state['user_input']))

            # Get the chatbot response
            response = chatbot(st.session_state['user_input'])

            # Append bot response to chat history
            st.session_state['chat_history'].append(('Bot', response))

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([st.session_state['user_input'], response, timestamp])

            # Clear the input field after processing    # This resets the input field
            st.session_state['user_input'] = ""  

    # User input field with on_change callback
    st.text_input("You:", key='user_input', on_change=on_user_input_change)

    # Display chat history
    for sender, message in reversed(st.session_state['chat_history']):
        alignment = 'right' if sender == "You" else 'left'
        st.markdown(
            f"<div style='text-align:{alignment};'><b>{sender}:</b> {message}</div>",
            unsafe_allow_html=True
        )


# Conversation History Menu
elif choice == "Conversation History":
    st.header("📚 Conversation History")
    
    if os.path.exists('chat_log.csv'):
        # Read the chat history using Pandas
        df = pd.read_csv('chat_log.csv')
        
        # Check if the CSV is empty
        if df.empty:
            st.warning("No conversation history available.")
        else:
            # Display the chat history in a dataframe
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No conversation history found. Start a chat to save interactions.")


# About Menu
elif choice == "About":
    st.write("The goal of the project is to create a chatbot that can understand and respond to user queries.")
    st.subheader("Project Overview:") 
    st.write(""" 
        The project is divided into two parts:
        1. NLP techniques and logistic regression algorithm are used to train the chatbot.
        2. The Streamlit framework is used to build a web application for the chatbot interface.
    """)
    st.subheader("Dataset:") 
    st.write(""" 
        The dataset used in this project is a collection of labelled intents and entities. 
        - Intents: The intents of the user input (example: "greeting", "budget", "about").
        - Entities: The entities extracted from user input (example: "hii", "How do I create a budget").
        - Text: The user input text.
    """)
    st.subheader("Streamlit Chatbot Interface")
    st.write("The chatbot interface is built using Streamlit. The interface includes a text data input field.")
    st.subheader("Conclusion")
    st.write("In this project, a chatbot is built that can understand and respond to user inputs.")

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ by Prem Kumar**")
