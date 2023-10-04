# Q&A Chatbot with Retrieval Augmented Generation (RAG)

Welcome to the Q&A Chatbot project! This repository contains the code and resources for building and deploying a Q&A chatbot that can answer questions based on a set of provided documents. The chatbot uses the Retrieval Augmented Generation (RAG) technique with Llama2 model to generate context-aware responses.

## Project Overview

The goal of this project is to create an interactive Q&A chatbot that can provide answers to user questions by searching through a collection of documents. The chatbot utilizes Llama2-13B model to understand user queries, retrieve relevant information from the documents, and generate informative responses. You can try with better models like Llama2-70B or Falcon-160B or also OpenAI's gtp-3.5-turbo.

## Features

- Interactive user interface for testing the chatbot.
- Context-aware question answering with the Llama2 model.
- Multiple document collection for answering questions.
- Requirements.txt file for easy dependency installation.

## Usage

To run the Q&A chatbot, follow these steps:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
3. Access the chatbot interface in your web browser (usually at http://localhost:8501).
4. Enter a question, and the chatbot will provide an answer based on the provided documents.

## Repo Structure:
The repository is organized as follows:

- **Q&A_chatbot.ipynb**: Jupyter Notebook explaining the code and build process.
- **app.py**: Contains the Streamlit app code for the chatbot user interface.
- **requirements.txt**: List of Python dependencies required for the project.
- **st_app.png**: Streamlit app UI demo image.

### NOTE:
- The streamlit app code can be done better and more optimised. Sooner, I will try to upload better version of it. Contributions are appreciated.

##Contributing

If you'd like to contribute to this project, please follow the standard GitHub flow:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

### Thank you
