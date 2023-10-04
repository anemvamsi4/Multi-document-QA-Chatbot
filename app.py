import streamlit as st
import transformers
from torch import cuda, bfloat16
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores  import FAISS
import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile
from streamlit_chat import message

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your documents"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def create_conversational_chain(vector_store,llm):

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(),
                                                 memory=memory)
    return chain

def display_chat_history(chain,prompt_template):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                prompt = prompt_template.format(user_prompt = user_input)
                output = conversation_chat(prompt, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")



def main():
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    with st.spinner("The model is loading. Upload your documents and please wait few minutes..."):
        model_id = 'NousResearch/Llama-2-7b-hf'

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )


        model_config = transformers.AutoConfig.from_pretrained(model_id)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )
        llm = HuggingFacePipeline(pipeline=generate_text)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embedding_model = HuggingFaceEmbeddings()

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embedding_model)

        # Create the chain object
        template = """You are a trained chatbot to answer questions from the texts given from some document. Give the answer to the prompt accurately and SHORTLY.
        INSTRUCTIONS:
        - Answer the questions from the context of document texts only.
        - When the question is in context just answer it and don't say I don't know
        - When the user prompt is not in the context given do not try to answer. Instead say "Sorry! I don't know about that. Ask me something in the context
        - Give explanatory answers for the given question from user prompt using the given texts
        - Also answer the questions by following the history
        --- You must FOLLOW all the INSTRUCTIONS given ---
        Promt: {user_prompt}
        """

        prompt_template = PromptTemplate(input_variables=['user_prompt'], template=template)
        chain = create_conversational_chain(vector_store,llm)

        display_chat_history(chain,prompt_template)

    else:
        st.error("Upload Documents to begin...")
if __name__ == "__main__":
    main()
