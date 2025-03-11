import os
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import arxiv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import numpy as np


def retrieve_paper_info(query: str, max_results: int):
    """
    Retrieve metadata for papers matching a query from arXiv.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "url": result.pdf_url
        })
    return papers

def download_pdf(url: str, save_dir: str = "document_store") -> str:
    """
    Download a PDF. Skip if the file already exists or is invalid.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, os.path.basename(urlparse(url).path))
    file_name += ".pdf"
    if os.path.exists(file_name):
        return file_name
    
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            if response.content[:4] != b"%PDF":
                return None
            with open(file_name, "wb") as f:
                f.write(response.content)
            return file_name
    except Exception as e:
        print(f"Failed to download {url}. Error: {str(e)}")
    return None

def download_pdfs_parallel(papers: list, max_workers: int = 20):
    """
    Use ThreadPoolExecutor to download PDFs in parallel.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_pdf, paper["url"], save_dir="document_store")
            for paper in papers
        ]
        results = [future.result() for future in futures]
    return results

def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a PDF file using PDFPlumberLoader.
    """
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    return documents

def extract_texts_parallel(pdf_paths: list, max_workers: int = 20):
    """
    Use ThreadPoolExecutor to extract text from PDFs in parallel.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_text_from_pdf, pdf_path) for pdf_path in pdf_paths]
        results = [future.result() for future in futures]
    flat_documents = [doc for sublist in results for doc in sublist]
    return flat_documents

def generate_embeddings(texts: list):
    """
    Generate embeddings for a list of texts using HuggingFaceEmbeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings.embed_documents(texts)

def generate_embeddings_parallel(texts: list, max_workers: int = 10):
    """
    Use ThreadPoolExecutor to generate embeddings in parallel.
    """
    chunk_size = len(texts) // max_workers
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_embeddings, chunk) for chunk in text_chunks]
        results = [future.result() for future in futures]
    return np.vstack(results)

def upload_batch(index, batch_vectors, batch_metadata):
    """
    Upload a single batch of vectors and metadata to Pinecone.
    """
    formatted_vectors = [
        {
            "id": f"vec_{i}",
            "values": vector,
            "metadata": meta
        }
        for i, (vector, meta) in enumerate(zip(batch_vectors, batch_metadata))
    ]
    if formatted_vectors:
        index.upsert(vectors=formatted_vectors)

def upload_to_pinecone_in_parallel(vectors: list, metadata: list, index, batch_size: int = 100, max_workers: int = 10):
    """
    Upload vectors and metadata to Pinecone in parallel batches.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            futures.append(executor.submit(upload_batch, index, batch_vectors, batch_metadata))
        
        # Wait for all batches to complete
        for future in futures:
            future.result()

def create_vectorDB(query: str, max_results: int):
    """
    Create a Pinecone vector database with metadata for papers matching a query.
    """
   
    papers = retrieve_paper_info(query, max_results)
    

    pdf_paths = download_pdfs_parallel(papers)
    
  
    documents = extract_texts_parallel(pdf_paths)
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
   
    texts = [doc.page_content for doc in chunks]
    embeddings = generate_embeddings_parallel(texts)
    
   
    metadata = []
    for paper in papers:
        metadata.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "summary": paper["summary"],
            "published": paper["published"],
            "url": paper["url"],
        })
    
   
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_name = "research-papers"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
    index = pc.Index(index_name)
    vectors = embeddings.tolist()
    
    # Upload in batches
    upload_to_pinecone_in_parallel(vectors, metadata, index)

   
