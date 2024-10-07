Website Content Conversational Chatbot
This repository contains the code for a Conversational Chatbot that interacts with users based on the content of a website. The chatbot scrapes website data, processes the text, and allows users to ask questions about the content, offering relevant responses in a conversational manner.

Project Overview
The chatbot uses the Google Gemini API to create contextual embeddings for website content, stores these embeddings using FAISS (Facebook AI Similarity Search), and enables interaction via a conversational model. The primary goal is to provide users with meaningful and context-driven responses to their questions, based on the contents of a website.

Key Features
Web Scraping: Extracts and processes website content using BeautifulSoup.
Text Embeddings: Utilizes Google Generative AI Embeddings to convert text into vectorized format.
Similarity Search: Implements FAISS to search through vectorized website content for relevant answers.
Conversational AI: The chatbot leverages Google Generative AI to generate responses in a natural, conversational style.
Streamlit Interface: Provides an intuitive web-based interface for users to interact with the chatbot.
How It Works
1. Scraping Website Content
The chatbot starts by scraping the website content entered by the user, extracting all relevant paragraphs using BeautifulSoup. This text is then split into manageable chunks for processing.

2. Text Embedding and Vector Store
The scraped website content is embedded using Google Generative AI Embeddings.
The vectorized text is stored using FAISS, allowing the chatbot to efficiently retrieve relevant sections of the website when answering user questions.
3. Conversational Chain
A conversational chain is generated using Google Gemini API to create a natural dialogue between the chatbot and the user.
The chatbot uses the context from the website, as well as previous questions and answers, to provide more informed responses.
4. User Interaction
Users input a website URL and ask questions based on the content.
The chatbot fetches relevant answers from the FAISS vector store and responds with contextually appropriate responses.
Project Setup
Prerequisites
Python 3.8 or higher
A Google API Key with access to Google Gemini API
Streamlit
FAISS
LangChain
BeautifulSoup
