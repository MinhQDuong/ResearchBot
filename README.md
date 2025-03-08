# ResearchBot
- A assistant chatbot for more precise research answers using Deepseek with RAG and Streamlit
- When answering questions about research papers. It is a well-known problem for chatbots to hallucinate or answer without proper source and citation, especially about the latest research topics.
- Using RAG and the latest LLM Deepseek, given an user query, this application downloads the most relevant papers, then use them to provide context and citations.
- This allows for more precise answers and users can access the sources to validate the answers and explore further.

## Resources used
- Libraries: langchain, pinecone, huggingface, streamlit
- Model used: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B