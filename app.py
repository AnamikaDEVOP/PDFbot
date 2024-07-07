import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import traceback

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to get LLM
@st.cache_resource
def get_llm():
    model_name = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=1024,  # Reduced back to 1024 for stability
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # Add this line
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def get_conversation_chain(vectorstore):
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True,
        max_tokens_limit=2048  # Add this line
    )
    return conversation_chain

def handle_userinput(user_question):
    st.write("Processing your question...")
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        st.write("Chat History:")
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"Human: {message.content}")
            else:
                st.write(f"AI: {message.content}")
        
        # Print additional debug information
        st.write("Debug Information:")
        st.write(f"Question: {user_question}")
        st.write(f"Answer: {response['answer']}")
        st.write(f"Source Documents: {response['source_documents']}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    st.write(f"Extracted {len(raw_text)} characters of text.")
                    
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks.")
                    
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Vector store created successfully.")
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
