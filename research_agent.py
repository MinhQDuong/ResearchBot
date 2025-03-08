import os
import torch
from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from process import (  
    generate_embeddings_parallel,
    create_vectorDB,
)
load_dotenv()
import nest_asyncio
nest_asyncio.apply()


st.markdown(
    """
    <style>
    /* Overall app styling */
    .stApp {
        background-color: #ffffff;  /* White background */
        color: #333333;
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #f8f9fa;  /* Light gray background */
        padding: 20px;
        border-right: 1px solid #e0e0e0;
    }

    .stSidebar h1 {
        color: #4a90e2;  /* Blue color for headings */
        font-size: 24px;
        margin-bottom: 20px;
    }

    .stSidebar h2 {
        color: #4a90e2;
        font-size: 20px;
        margin-bottom: 15px;
    }

    .stSidebar .stFileUploader > div {
        background-color: #ffffff;
        border: 2px dashed #4a90e2;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    .stSidebar .stFileUploader > div:hover {
        background-color: #f0f8ff;  /* Light blue on hover */
        border-color: #357abd;
    }

    .stSidebar .stButton > button {
        background-color: #4a90e2;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
        width: 100%;
    }

    .stSidebar .stButton > button:hover {
        background-color: #357abd;  /* Darker blue on hover */
    }

    /* Chat message styling */
    .stChatMessage {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 80%;
    }

    /* User message styling */
    .stChatMessage.user {
        background-color: #4a90e2;  /* Blue background for user */
        color: white;               /* White text for user */
        margin-left: auto;
        margin-right: 0;
    }

    /* Assistant message styling */
    .stChatMessage.assistant {
        background-color: #e0e0e0;  /* Light gray background for assistant */
        color: #333333;             /* Dark text for assistant */
        margin-left: 0;
        margin-right: auto;
    }

    /* Avatar styling */
    .stChatMessage .avatar {
        background-color: #4a90e2;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    /* Spinner styling */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4a90e2;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #4a90e2;
    }

    /* Citation styling */
    .citation {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }

    .citation h4 {
        color: #4a90e2;
        margin-bottom: 5px;
    }

    .citation p {
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Define the index name
index_name = "research-papers"


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",  
        spec=ServerlessSpec(
            cloud="aws",  
            region="us-east-1" 
        )
    )
index = pc.Index(index_name)

# Load DeepSeek 1.5B model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    generator = pipeline(
        "text-generation",
        model=model,
        device=0 if device == "cuda" else -1,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    return generator

generator = load_model()

# Define the prompt template
PROMPT_TEMPLATE = """
You are an expert research assistant. It is optional to use the provided context to help answer the query. 
Give the answer in a concise way. If unsure, state that you don't know.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Initialize the language model
LANGUAGE_MODEL = HuggingFacePipeline(pipeline=generator)

# Function to generate answers
def generate_answer(user_query, context_documents):
    """
    Generate an answer using the retrieved context documents.
    """
    # Extract the text from the metadata
    context_text = "\n\n".join([doc['metadata'].get('title', 'N/A') for doc in context_documents])
    
    # Create the prompt and response chain
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    
    # Invoke the response chain
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    
    # Post-process the response to extract only the answer
    if isinstance(response, str):
        if "Answer:" in response:
            return response.split("Answer:")[1].strip()
        return response
    elif isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "")
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[1].strip()
        return generated_text
    else:
        return "No answer generated."

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# App title
st.title("📘 ResearchBot")
st.markdown("### Your Intelligent Research Assistant")
st.markdown("---")



# Sidebar for PDF upload
with st.sidebar:
    st.markdown("<h1>Document Upload</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        help="Select a PDF document for analysis",
    )

    

        # Show the spinner while uploading
        
    if uploaded_file:
        # Save the uploaded file
        save_dir = "document_store"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from the uploaded PDF
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings for the chunks
        texts = [doc.page_content for doc in chunks]
        embeddings_upload = generate_embeddings_parallel(texts)
        
        
        vectors = embeddings_upload.tolist()
        metadata = [doc.metadata for doc in chunks]
        vectors_with_ids = [
                    {"id": f"vec_{i}", "values": vector}
                    for i, vector in enumerate(vectors)
                ]
        index.upsert(vectors=vectors_with_ids)

        
        
        st.success("✅ PDF uploaded successfully! Ask your questions.")
        st.session_state.upload_complete = True
        
            

   

    st.markdown("<h2>Actions</h2>", unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        document_store_path = "document_store"  # Update this path
        for filename in os.listdir(document_store_path):
            file_path = os.path.join(document_store_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting file {filename}: {e}")
        st.success("Conversation history and vector database cleared!")

# Chat interface
user_input = st.text_input("Ask a question about research papers:")

if user_input:
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)
    with st.spinner("Getting the most relevant papers..."):
        # Generate query embedding
        query_embedding = embeddings.embed_query(user_input)

        
        create_vectorDB(user_input, max_results=3)
            
        
        # Query Pinecone index
        query_response = index.query(
            vector=query_embedding,  
            top_k=3,               
            include_metadata=True    
        )
        
        # Extract the metadata from the response
        context_documents = []
        for match in query_response.matches:
            context_documents.append({
                
                "metadata": match.metadata  # Include metadata for citations
            })
        
    # Generate answer with a spinner
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_input, context_documents)
    
    # Display assistant message
    with st.chat_message("assistant", avatar="🤖"):
        st.write(answer)
    st.session_state.conversation_history.append((user_input, answer))
    
    # Display citations
    st.write("**Citations:**")
    for doc in context_documents:
        metadata = doc["metadata"]
        if metadata == None:
            continue
        st.write(f"- **Title:** {metadata.get('title', 'N/A')}")
        st.write(f"  **Authors:** {', '.join(metadata.get('authors', []))}")
        st.write(f"  **Published:** {metadata.get('published', 'N/A')}")
        st.write(f"  **Link:** {metadata.get('url', 'N/A')}")
        st.write("---")


def clip_answer(answer, max_length=200):
        
        if len(answer) > max_length:
            return answer[:max_length] + "..."
        return answer
# Display conversation history
st.header("Conversation History")
for i, (question, answer) in enumerate(st.session_state.conversation_history):
    with st.chat_message("user"):
        st.write(f"**Q{i+1}:** {question}")
    with st.chat_message("assistant", avatar="🤖"):
        st.write(f"**A{i+1}:** {clip_answer(answer)}")
    st.write("---")