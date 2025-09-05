## End to End RAGops ( Not Finish ) 

### Continuous improvement

Adding : 

- Logging ( Elastic Stack + Filebeat )  + observability ( Kibana )
- Sematic Caching ( Redis ) 
- Model Serving ( BentoML instead of FastAPI for dynamic batch request + faster inference & embedding by leverage CPU/GPU )
- Deploy on Kubernetes or Cloud-Run
- Rare-limit design + Load balancing
- Adding LLM guardrails 


### Data-Pipeline  : Ingestion from arXiv API -> Google Cloud (data-lake) -> Google BigQuery (data-warehouse) 



#### Airflow Pipeline 
- Using airflow to set up schedule data pipeline from arXiv API -> Google Cloud 
![alt text](images/image.png) 


#### Google Cloud Storage (Data Lake - Gold layers)
1/ Big Load & Monthly Load from arXiv API pipeline -> Storage 
![alt text](images/storage.png) 

- The cs_date_date.parquet is the chunk loading from ( 2024-01-01 - 2025-09-01)
- The cs_latest.parquet or econ_latest.parquet are monthly recurring load 



#### Google BigQuery (Data Warehose - Silver layers) 
1/ Data Transfer From Cloud. Storage to update monthly 
![alt text](images/image-2.png)

2/ Schedule ETL Queries ( remove duplicate fixing datetime)
![alt text](images/image-3.png)

3/ Table view 
![alt text](images/image-1.png)



### Model-Pipeline for RAG system 

- **Data Ingestion**: Retrieves research paper data from Google BigQuery ( Optional : adding airflow from Google Big Query -> Milvus ( Vector DB ) every month  or run on bash:  curl -s -X POST http://localhost:3030/ingest | jq ( adding new monthly data in ) ) 

- **Vector Storage**: Uses Milvus as a vector database
- **Embedding**: Uses OpenAI's text-embedding-3-small 
- **Retrieval**: Fast vector similarity search using Milvus
- **Reranking**: Semantic reranking with Cohere
- **Answer Generation**: Synthesizes information with OpenAI's GPT-4.1
- **Evaluation**: Automatic evaluation of responses for hallucination, truthfulness, accuracy, and relevancy
- **Metadata Enrichment**: Extracts additional paper metadata from URLs 

#### Example Query & Response
Accessing : localhost:3030/docs/query

Below is a real summarize-mode example (query starts with `summarize ... (pdf_url)`), showing unified retrieval + rerank + summarization + LLM evaluation output structure.

Request (POST /query):
```json
{
	"query": "summarize Empowering Biomedical Discovery with AI Agents (http://arxiv.org/pdf/2404.02831v2)",
	"top_k": 5,
	"get_metadata": false
}
```

Response (truncated for readability):
```json
{
  "query": "summarize Empowering Biomedical Discovery with AI Agents (http://arxiv.org/pdf/2404.02831v2)",
  "answer": "1. Problem\nThe paper discusses the potential of artificial intelligence (AI) agents in accelerating biomedical discovery. The authors argue that while current AI models are proficient in analyzing large datasets and automating repetitive tasks, they lack the ability to generate novel hypotheses, a critical aspect of scientific discovery.",
  "papers": [
    {
      "id": "doc_a5fb9ea81abeb8365893ea1ca20e8ec0",
      "title": "Empowering Biomedical Discovery with AI Agents",
      "authors": "",
      "abstract": "",
      "url": "",
      "score": 0.99997437,
      "metadata": null
    },
    {
      "id": "doc_ac5470ddb78ce490b229f543241f0286",
      "title": "Discovery of Disease Relationships via Transcriptomic Signature Analysis   Powered by Agentic AI",
      "authors": "",
      "abstract": "",
      "url": "",
      "score": 0.70069003,
      "metadata": null
    },
    {
      "id": "doc_1fcd308a1daeb8bceb477f73aeb35d4b",
      "title": "Autonomous Artificial Intelligence Agents for Clinical Decision Making   in Oncology",
      "authors": "",
      "abstract": "",
      "url": "",
      "score": 0.42518753,
      "metadata": null
    },
    {
      "id": "doc_2f654c21d5e71a73e91b6ce1055834e8",
      "title": "Toward Safe Evolution of Artificial Intelligence (AI) based   Conversational Agents to Support Adolescent Mental and Sexual Health   Knowledge Discovery",
      "authors": "",
      "abstract": "",
      "url": "",
      "score": 0.416856,
      "metadata": null
    },
    {
      "id": "doc_52ef713ee347a713e3d3d31edcd2c7e9",
      "title": "AI Agent Behavioral Science",
      "authors": "",
      "abstract": "",
      "url": "",
      "score": 0.3431216,
      "metadata": null
    }
  ],
  "evaluation": {
    "hallucination_score": 5,
    "truthfulness_score": 5,
    "accuracy_score": 5,
    "relevancy_score": 10,
    "explanation": "1. Hallucination Score: 5\n   Explanation: The generated answer seems to provide a detailed summary of the paper \"Empowering Biomedical Discovery with AI Agents\". However, without the actual content of the paper or its abstract, it's impossible to verify if all the information in the answer is directly supported by the documents. Therefore, a neutral score of 5 is given.\n\n2. Truthfulness Score: 5\n   Explanation: Similar to the hallucination score, without the actual content of the paper or its abstract, it's impossible to verify the factual correctness of the information in the answer. Therefore, a neutral score of 5 is given.\n\n3. Accuracy Score: 5\n   Explanation: The accuracy of the answer in terms of correctly interpreting and representing the information from the documents cannot be determined due to the lack of content in the retrieved documents. Therefore, a neutral score of 5 is given.\n\n4. Relevancy Score: 10\n   Explanation: The answer is highly relevant to the user's query. The user asked for a summary of the paper \"Empowering Biomedical Discovery with AI Agents\", and the generated answer provides a detailed summary of the paper, covering various aspects such as the problem, methods, key results, interpretation/significance, and limitations & future work.\n\nOverall Assessment: The generated answer seems to be a comprehensive and well-structured summary of the paper \"Empowering Biomedical Discovery with AI Agents\". However, due to the lack of content in the retrieved documents, it's impossible to verify the hallucination, truthfulness, and accuracy of the information in the answer. The answer is highly relevant to the user's query."
  },
  "summary": "1. Problem\nThe paper discusses the potential of artificial intelligence (AI) agents in accelerating biomedical discovery. The authors argue that while current AI models are proficient in analyzing large datasets and automating repetitive tasks, they lack the ability to generate novel hypotheses, a critical aspect of scientific discovery.\n\n2. Methods\nThe authors propose a new approach where AI agents are integrated with human expertise, large language models (LLMs), machine learning (ML) tools, and experimental platforms to form a compound AI system. These AI agents are designed to formulate biomedical hypotheses, evaluate them critically, characterize their uncertainty, and use this information to refine their scientific knowledge bases.\n\n3. Key Results\nThe authors suggest that AI agents can impact various areas of biomedical research, including virtual cell simulation, programmable control of phenotypes, and the design of cellular circuits. They can also assist in developing new therapies by predicting the effects of genetic modifications or drug treatments on cell behavior.\n\n4. Interpretation / Significance\nThe integration of AI agents in biomedical research could lead to significant advancements in the field. By combining human creativity and expertise with AI's analytical capabilities, these agents can accelerate discovery workflows, making them faster and more resource-efficient. They can also provide insights that might not have been possible using ML alone.\n\n5. Limitations & Future Work\nThe authors acknowledge that there are ethical considerations and challenges associated with the use of AI agents in biomedical research. These include the potential for harm if AI agents are allowed to make changes in environments through ML tools or calls to experimental platforms. There is also a need for large experimental datasets that cover diverse use cases beyond the current focus on a few biomedical domains. Future work should focus on addressing these challenges and ensuring responsible implementation of AI agents in biomedical research. Note: Source text appears partial/truncated.",
  "ranking": [
    {
      "rank": 1,
      "id": "doc_a5fb9ea81abeb8365893ea1ca20e8ec0",
      "title": "Empowering Biomedical Discovery with AI Agents",
      "score": 0.99997437
    },
    {
      "rank": 2,
      "id": "doc_ac5470ddb78ce490b229f543241f0286",
      "title": "Discovery of Disease Relationships via Transcriptomic Signature Analysis   Powered by Agentic AI",
      "score": 0.70069003
    },
    {
      "rank": 3,
      "id": "doc_1fcd308a1daeb8bceb477f73aeb35d4b",
      "title": "Autonomous Artificial Intelligence Agents for Clinical Decision Making   in Oncology",
      "score": 0.42518753
    },
    {
      "rank": 4,
      "id": "doc_2f654c21d5e71a73e91b6ce1055834e8",
      "title": "Toward Safe Evolution of Artificial Intelligence (AI) based   Conversational Agents to Support Adolescent Mental and Sexual Health   Knowledge Discovery",
      "score": 0.416856
    },
    {
      "rank": 5,
      "id": "doc_52ef713ee347a713e3d3d31edcd2c7e9",
      "title": "AI Agent Behavioral Science",
      "score": 0.3431216
    }
  ]
}
```

Key fields:
- `answer`: Concise 1–2 section opening lines for quick preview.
- `summary`: Full structured multi-section summary (Problem, Methods, Key Results, Interpretation, Limitations & Future Work).
- `papers`: Top reranked context documents (Milvus vector search → Cohere rerank) with relevance scores.
- `ranking`: Explicit ordered list mirroring `papers` for lightweight consumers.
- `evaluation`: Automatic LLM-as-judge quality assessment (hallucination, truthfulness, accuracy, relevancy) on the scale of 0 (worst) - 10 (best) .

Notes:
- When no abstract is stored, system fetches PDF (arXiv or direct .pdf) and summarizes first pages (truncated) with a truncation note.
- Incremental ingestion (monthly) reuses deterministic IDs; unchanged papers are skipped.
- Summarize mode is triggered by leading `summarize` keyword in the query. 
