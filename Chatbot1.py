import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from urllib.parse import urljoin

# Configure Google Gemini API
# Put your API key
os.environ["GOOGLE_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to scrape website content
def parse_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Scrape text content
        parsed_text = ' '.join([p.text for p in soup.find_all('p')])  # Scrape paragraph text
        
        # Scrape image URLs
        images = []
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url:
                full_url = urljoin(url, img_url)  # Convert to absolute URL
                images.append(full_url)
                
        return parsed_text, images
    except Exception as e:
        st.error(f"Error parsing website: {e}")
        return None, []

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to generate the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are having a conversation with a user about the following topic.
    Build on the previous questions and answers, and keep the conversation flowing naturally.
    
    If the answer is not in the provided context, make an educated guess. If the question is completely unrelated to the context, say: 
    "I can't answer that from the provided information."

    Previous Conversation: 
    {previous_chat}
    
    Context: {context}
    
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)  # Increase temperature for more creativity
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "previous_chat", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user question and perform similarity search
# Function to handle user question and perform similarity search
# Updated function to generate the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are having a conversation with the user about a topic based on the website content.
    Continue the conversation by building on previous answers and questions.
    Use the website context to answer the user's question, but also keep in mind the previous conversation.
    
    If the current question is related to previous questions, provide a response that builds on the previous answers.
    If the answer cannot be derived from the context, say: "I can't answer that from the provided information."

    Previous Conversation:
    {previous_chat}

    Context from Website:
    {context}

    User's Question:
    {question}

    Your Response:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "previous_chat", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user question and perform similarity search only on website content
# Function to handle user question and perform similarity search only on website content
# Function to handle user question and perform similarity search only on website content
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search based on user question, only on the website content
    relevant_docs = vector_store.similarity_search(user_question, k=3)  # Return more chunks for richer context
    
    # If no relevant documents found, use the whole parsed content as a fallback
    if not relevant_docs and st.session_state['parsed_content']:
        st.warning("No specific matches found, using entire website content as fallback.")
        relevant_docs = [Document(page_content="\n".join(st.session_state['parsed_content']))]

    chain = get_conversational_chain()

    # Combine previous questions and answers for context but do not apply FAISS on it
    previous_chat = "\n".join(
        [f"Q: {q}\nA: {r}" for q, r in zip(st.session_state['questions'], st.session_state['responses'])]
    )

    # Use relevant website context (retrieved by FAISS) and previous chat for conversational flow
    response = chain({
        "input_documents": relevant_docs,  # Correctly pass documents here
        "context": "\n".join([doc.page_content for doc in relevant_docs]),  # Only use website content for FAISS
        "previous_chat": previous_chat,  # Previous chat history, not part of FAISS
        "question": user_question
    }, return_only_outputs=True)

    # Store the question and response in session state
    st.session_state['questions'].append(user_question)
    st.session_state['responses'].append(response["output_text"])

    st.write("Reply: ", response["output_text"])

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Website Content Chat", layout="wide")
    st.header("Explore and Discuss Website Content Using Chatbot")

    # Initialize session state for storing questions and responses
    if 'questions' not in st.session_state:
        st.session_state['questions'] = []
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []

    # Input for website URL
    website_url = st.text_input("Enter the website URL")

    # Parse website content and store in session state if not already parsed
    if website_url and 'parsed_content' not in st.session_state:
        parsed_text, image_urls = parse_website(website_url)
        if parsed_text:
            text_chunks = get_text_chunks(parsed_text)
            st.session_state['parsed_content'] = text_chunks  # Store chunks in session state
            st.session_state['image_urls'] = image_urls       # Store image URLs in session state
            get_vector_store(text_chunks)
            st.success("Website content parsed and vectorized.")

    # Display images if available
    if 'image_urls' in st.session_state and st.session_state['image_urls']:
        st.subheader("Images Found on Website")
        for img_url in st.session_state['image_urls']:
            st.image(img_url)

    # Chat functionality here (as before)
    if 'parsed_content' in st.session_state:
        user_question = st.text_input("Ask a question about the website content")  # Enable chat input
        if user_question:
            user_input(user_question)
    else:
        st.info("Please enter a website link and wait for it to be parsed before asking questions.")

    # Display all saved questions and responses
    if st.session_state['questions']:
        st.subheader("Previous Questions and Responses")
        for i, (question, response) in enumerate(zip(st.session_state['questions'], st.session_state['responses'])):
            st.write(f"**Question {i+1}:** {question}")
            st.write(f"**Response {i+1}:** {response}")
            st.write("---")

if __name__ == "__main__":
    main()
