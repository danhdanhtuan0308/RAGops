
## Features & Pipeline for RAG system 

- **Data Ingestion**: Retrieves research paper data from Google BigQuery
- **Vector Storage**: Uses Milvus as a vector database
- **Embedding**: Uses OpenAI's text-embedding-3-small 
- **Retrieval**: Fast vector similarity search using Milvus
- **Reranking**: Semantic reranking with Cohere
- **Answer Generation**: Synthesizes information with OpenAI's GPT-4.1
- **Evaluation**: Automatic evaluation of responses for hallucination, truthfulness, accuracy, and relevancy
- **Metadata Enrichment**: Extracts additional paper metadata from URLs 
- **Implement Cache** : Saving cost + time 