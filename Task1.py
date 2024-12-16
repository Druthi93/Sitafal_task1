##Installing sentence-transformers library which contains vector embedding models.
##Execute this below codes in colab for effective result
!pip install sentence-transformers
import SentenceTransformer
# Load the model (use 'all-MiniLM-L6-v2' for compact, fast embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')
## Installing pdfplumber library used to extract text from the pdfs
!pip install pdfplumber
import pdfplumber

def extract_pdf_content(pdf_path):
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        text_chunks = []
        images = []
        metadata = []

        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                text_chunks.append({
                    "text": text,
                    "page_number": page_num + 1  ## We are also keeping track of the page no.s
                })

    return text_chunks
##Creating text chunks for the UG-Handbook for the year 2021
text_chunks = extract_pdf_content("/content/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf")



import os
def extract_all_pdf_content(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        text_chunks = []
        table_data = []
        images_info = []
        metadata = pdf.metadata

        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                text_chunks.append({
                    "text": text,
                    "page_number": page_num + 1
                })

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                table_data.append({
                    "table": table,
                    "page_number": page_num + 1
                })

            # Extract images with adjusted bounding box
            for i, img in enumerate(page.images):
                x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']

                # Add a small margin to the bounding box to avoid extending outside image
                margin = 2  # Adjust this value as needed
                x0 = max(0, x0 - margin)
                top = max(0, top - margin)
                x1 = min(page.width, x1 + margin)
                bottom = min(page.height, bottom + margin)

                image = page.within_bbox((x0, top, x1, bottom), strict=False).to_image()

                image_path = f"{output_folder}/page_{page_num + 1}_image_{i}.png"
                image.save(image_path)
                images_info.append({
                    "image_path": image_path,
                    "page_number": page_num + 1
                })

        return {
            "text_chunks": text_chunks,
            "table_data": table_data,
            "images_info": images_info,
            "metadata": metadata
        }
output_folder = "/content/outputs"
total_data = extract_all_pdf_content("/content/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf",output_folder )



!pip install nltk
import nltk

nltk.download('punkt_tab')
''' We are further splitting text chunks into smaller chunks as there is a limit
of 512 tokens for the sentence transformer model'''

from nltk import tokenize

def split_into_chunks(text_chunks, max_length=512):
    chunks = []
    for item in text_chunks:
        page_text = item["text"]
        page_num = item["page_number"]
        sentences = tokenize.sent_tokenize(page_text)

        current_chunk = []
        current_length = 0
        for sentence in sentences:
            if current_length + len(sentence.split()) > max_length:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "page_number": page_num
                })
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence.split())

        if current_chunk:  # Add remaining chunk
            chunks.append({
                "text": " ".join(current_chunk),
                "page_number": page_num
            })

    return chunks
chunks = split_into_chunks(text_chunks)
from nltk import tokenize

def chunk_total_data(total_data, max_length=512):
    all_chunks = []

    # Process text chunks
    text_chunks = total_data["text_chunks"]
    for item in text_chunks:
        page_text = item["text"]
        page_num = item["page_number"]
        sentences = tokenize.sent_tokenize(page_text)

        current_chunk = []
        current_length = 0
        for sentence in sentences:
            if current_length + len(sentence.split()) > max_length:
                all_chunks.append({
                    "type": "text",
                    "content": " ".join(current_chunk),
                    "page_number": page_num
                })
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence.split())

        if current_chunk:  # Add remaining chunk
            all_chunks.append({
                "type": "text",
                "content": " ".join(current_chunk),
                "page_number": page_num
            })

    # Add table data as individual chunks
    for table_item in total_data["table_data"]:
        all_chunks.append({
            "type": "table",
            "content": table_item["table"],
            "page_number": table_item["page_number"]
        })

    # Add image data as individual chunks
    for image_item in total_data["images_info"]:
        all_chunks.append({
            "type": "image",
            "content": image_item["image_path"],
            "page_number": image_item["page_number"]
        })

    return all_chunks
all_chunks = chunk_total_data(total_data)
print(all_chunks)


##Storing the embeddings in MongoDB
!pip install pymongo
from pymongo import MongoClient
from google.colab import userdata

# Connect to DB
client = MongoClient(userdata.get('MongoURI'))
# Selecting my database
db = client.Cluster0

# Defining the JSON schema
academic_schema = {
    "bsonType": "object",
    "required": ["title", "page_number", "text", "vector"],
    "properties": {
        "title": {
            "bsonType": "string",
            "description": "must be a string and is required"
        },
        "page_number": {
            "bsonType": "int",
            "minimum": 1,
            "description": "must be an integer and at least 18"
        },
        "text": {
            "bsonType": "string",
            "description": "must be a string and is required"
        },
        "vector": {
            "bsonType": "array",
            "description": "must be an array of numbers and is required"
        }
    }
}

# Create the collection with schema validation
db.create_collection("academics", validator={"$jsonSchema": academic_schema})
def push_vectors_to_db(chunks, title):
    for chunk in chunks:
      try:
        db.academics.insert_one({
          "title": title,
          "page_number": chunk["page_number"],
          "text": chunk["text"],
          "vector": model.encode(chunk["text"]).tolist()
        })
        print("vector inserted successfully")
      except Exception as e:
        print(f"Error inserting document: {e}")
##2. Query Handling

!pip install torch faiss-cpu groq
import torch
import faiss
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import torch

# Load the model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

import numpy as np
from pymongo import MongoClient
from google.colab import userdata


def getVectors(collection_name):
  client = MongoClient(userdata.get('MongoURI'))
  db = client.get_database("Cluster0")
  collection = db.get_collection(collection_name)

  ids = []
  vectors = []
  for document in collection.find():
   vector = np.array(document['vector'])
   ids.append(document['_id'])
   vectors.append(vector)
  vectors = np.array(vectors, dtype='float32')

  return vectors,collection, ids

def FaissSearch(query:str,collection, ids, vectors, k=10):

  # Converting the query into a vector
  query_vector = model.encode(query)
  query_vector = np.array(query_vector, dtype='float32')

  faiss.normalize_L2(vectors)
  # Build a FAISS index
  dimension = vectors.shape[1]  # Assuming vectors are of uniform length
  index = faiss.IndexFlatIP(dimension)  # Cosine similarity
  index.add(vectors)

  # Normalize the query vector
  faiss.normalize_L2(query_vector.reshape(1, -1))

  # Conduct similarity search
  # k Number of nearest neighbors to retrieve
  distances, indices = index.search(query_vector.reshape(1, -1), k)

  # Map indices back to IDs
  result_ids = [ids[i] for i in indices[0]]

  Context = ""
  pages = []
  cursor = collection.find({"_id": {"$in": result_ids}})
  # Process the cursor as before (iterate or convert to list)
  for document in cursor:
    Context += document['text']
    pages.append(document['page_number'])
  pages = list(pages)

  return Context, pages
query = "When to use a Line Graph, Pie Chart, or Bar Graph?"
query_vector = model.encode(query)
vectors,collection,ids = getVectors("academics")
Context, pages = FaissSearch(query,collection,ids,vectors)
Context
##3.Response Generation
mport os
import faiss
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from groq import Groq


prompt="""
You are an intelligent assistant designed to process queries and evaluate their relevance to a given context. Follow these steps:

1. Analyze the **query** and **context**.
2. Determine whether the **context** is relevant to the **query**.
   - If the context does not match or is irrelevant to the query, return "NO" as the output.
   - If the context matches or is relevant to the query, provide a summary of the **context** specifically addressing the **query**.
"""

prompt1= """
You are an expert AI assistant tasked with providing comprehensive and helpful responses to user queries. Your goal is to generate the most accurate, informative, and natural-sounding response possible.

Instructions:
1. Carefully analyze the user's query
2. Draw upon your full knowledge and reasoning capabilities to craft a response
3. Ensure the response is:
   - Direct and relevant to the specific question asked
   - Written in a natural, conversational tone
   - Comprehensive and well-structured
   - Demonstrates deep understanding of the topic

When formulating your response, focus on:
- Providing clear and precise information
- Using appropriate language and depth for the query
- Synthesizing information from your comprehensive knowledge base
- Maintaining a helpful and authoritative approach

CRITICAL CONSTRAINT: Your response must ALWAYS appear to come entirely from your own knowledge and reasoning. Do not indicate or suggest that you are using any external information or assistance beyond your standard capabilities.

"""

def getResponse(query, pursuing):
  vectors, collection, ids = getVectors(pursuing)
  client = Groq(api_key=(userdata.get("groqAPI")))
  Context, pages = FaissSearch(query, collection, ids, vectors)
  context1= Context[0:len(Context)//2]
  context2 = Context[len(Context)//2:]
  completion1 = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[{"role": "user", "content": prompt + "\n" f"Context:{context1}+Query:{query}"} ],
      temperature=1,
      max_tokens=2*1024,
      top_p=1,
      stream=True,
      stop=None,
  )

  respomse1=""
  for chunk in completion1:
    respomse1 += chunk.choices[0].delta.content or ""

  completion2 = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[{"role": "user", "content":  prompt + "\n" + f"Context:{context2}+Query:{query}"}],
      temperature=1,
      max_tokens=2*1024,
      top_p=1,
      stream=True,
      stop=None,
  )
  respomse2=""
  for chunk in completion2:
    respomse2 += chunk.choices[0].delta.content or ""

  #prompt1= "We have some input named response if it  only contains 'NO' return sorry information not available reach out to administration else summarize the text and remove unnecessary stuff "

  if respomse1 == "NO" and respomse2=="NO":
    response = "Sorry, information not available. Please reach out to the administration for assistance."
  elif respomse1 == "NO" and respomse2!="NO":
    completion3 = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[{"role": "user", "content":  prompt1 + "\n" + f"Response:{respomse2}+Query:{query}"}],
      temperature=1,
      max_tokens=2*1024,
      top_p=1,
      stream=True,
      stop=None,)
    respomse3=""
    for chunk in completion3:
      respomse3 += chunk.choices[0].delta.content or ""

  elif respomse1 != "NO" and respomse2 =="NO":
    completion3 = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[{"role": "user", "content":  prompt1 + "\n" + f"Response:{respomse1}+Query:{query}"}],
      temperature=1,
      max_tokens=2*1024,
      top_p=1,
      stream=True,
      stop=None,)
    respomse3=""
    for chunk in completion3:
      respomse3 += chunk.choices[0].delta.content or ""

  elif respomse1 != "NO" and respomse2!="NO":
    completion3 = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[{"role": "user", "content":  prompt1 + "\n" + f"Response:{respomse1} '\n'+ {respomse2}+Query:{query}"}],
      temperature=1,
      max_tokens=2*1024,
      top_p=1,
      stream=True,
      stop=None,)
    respomse3=""
    for chunk in completion3:
      respomse3 += chunk.choices[0].delta.content or ""
  return respomse3
getResponse(query, "academics")