The above is just the codes for task 1
For complete task below colab link will be the execution.
https://colab.research.google.com/drive/15b-JTlUi0gEBOkEXZHzcWVNKYB12bBjh?usp=sharing


Task Details: 
Task 1: Chat with PDF Using RAG Pipeline
Overview
The goal is to implement a Retrieval-Augmented Generation (RAG) pipeline that allows users to
interact with semi-structured data in multiple PDF files. The system should extract, chunk,
embed, and store the data for eFicient retrieval. It will answer user queries and perform
comparisons accurately, leveraging the selected LLM model for generating responses.
Functional Requirements
1. Data Ingestion
• Input: PDF files containing semi-structured data.
• Process:
o Extract text and relevant structured information from PDF files.
o Segment data into logical chunks for better granularity.
o Convert chunks into vector embeddings using a pre-trained embedding model.
o Store embeddings in a vector database for eFicient similarity-based retrieval.

2. Query Handling
• Input: User's natural language question.
• Process:
o Convert the user's query into vector embeddings using the same embedding
model.
o Perform a similarity search in the vector database to retrieve the most relevant
chunks.
o Pass the retrieved chunks to the LLM along with a prompt or agentic context to
generate a detailed response.

3. Comparison Queries
• Input: User's query asking for a comparison
• Process:
o Identify and extract the relevant terms or fields to compare across multiple PDF
files.
o Retrieve the corresponding chunks from the vector database.
o Process and aggregate data for comparison.
o Generate a structured response (e.g., tabular or bullet-point format).

4. Response Generation
• Input: Relevant information retrieved from the vector database and the user query.
• Process:
o Use the LLM with retrieval-augmented prompts to produce responses with exact
values and context.
o Ensure factuality by incorporating retrieved data directly into the response.

Example Data:

https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-
presentations/tables-charts-and-graphs-with-examples-from.pdf

Extract accurate information:
1. From page 2 get the exact unemployment information based on type of degree input
2. From page 6 get the tabular data
