#!/usr/bin/env python
# coding: utf-8

# # Basic RAG (Retrieval Augmented Generation)

# In[ ]:


# ! pip install faiss-cpu "mistralai>=0.1.2"


# ### Load API key

# In[1]:


from helper import load_mistral_api_key
api_key, dlai_endpoint = load_mistral_api_key(ret_key=True)


# ### Get data
# 
# - You can go to https://www.deeplearning.ai/the-batch/
# - Search for any article and copy its URL.

# ### Parse the article with BeautifulSoup 

# In[2]:


import requests
from bs4 import BeautifulSoup
import re

response = requests.get(
    "https://www.deeplearning.ai/the-batch/a-roadmap-explores-how-ai-can-detect-and-mitigate-greenhouse-gases/"
)
html_doc = response.text
soup = BeautifulSoup(html_doc, "html.parser")
tag = soup.find("div", re.compile("^prose--styled"))
text = tag.text
print(text)


# ### Optionally, save the text into a text file
# - You can upload the text file into a chat interface in the next lesson.
# - To download this file to your own machine, click on the "Jupyter" logo to view the file directory.  

# In[3]:


file_name = "AI_greenhouse_gas.txt"
with open(file_name, 'w') as file:
    file.write(text)


# ### Chunking

# In[4]:


chunk_size = 512
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# In[5]:


len(chunks)


# ### Get embeddings of the chunks

# In[6]:


import os
from mistralai.client import MistralClient


def get_text_embedding(txt):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    embeddings_batch_response = client.embeddings(model="mistral-embed", input=txt)
    return embeddings_batch_response.data[0].embedding


# In[9]:


import numpy as np

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])


# In[10]:


text_embeddings


# In[11]:


len(text_embeddings[0])


# ### Store in a vector databsae
# - In this classroom, you'll use [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

# In[12]:


import faiss

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)


# ### Embed the user query

# In[13]:


question = "What are the ways that AI can reduce emissions in Agriculture?"
question_embeddings = np.array([get_text_embedding(question)])


# In[14]:


question_embeddings


# ### Search for chunks that are similar to the query

# In[15]:


D, I = index.search(question_embeddings, k=2)
print(I)


# In[16]:


retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
print(retrieved_chunk)


# In[19]:


prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""


# In[20]:


from mistralai.models.chat_completion import ChatMessage


def mistral(user_message, model="mistral-small-latest", is_json=False):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    messages = [ChatMessage(role="user", content=user_message)]

    if is_json:
        chat_response = client.chat(
            model=model, messages=messages, response_format={"type": "json_object"}
        )
    else:
        chat_response = client.chat(model=model, messages=messages)

    return chat_response.choices[0].message.content


# In[21]:


response = mistral(prompt)
print(response)


# In[ ]:





# ## RAG + Function calling

# In[22]:


def qa_with_context(text, question, chunk_size=512):
    # split document into chunks
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    # load into a vector database
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
    # create embeddings for a question
    question_embeddings = np.array([get_text_embedding(question)])
    # retrieve similar chunks from the vector database
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    # generate response based on the retrieve relevant text chunks

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    response = mistral(prompt)
    return response


# In[23]:


I.tolist()


# In[24]:


I.tolist()[0]


# In[25]:


import functools

names_to_functions = {"qa_with_context": functools.partial(qa_with_context, text=text)}


# In[26]:


tools = [
    {
        "type": "function",
        "function": {
            "name": "qa_with_context",
            "description": "Answer user question by retrieving relevant context",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "user question",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


# In[ ]:


question = """
What are the ways AI can mitigate climate change in transportation?
"""

client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)

response = client.chat(
    model="mistral-large-latest",
    messages=[ChatMessage(role="user", content=question)],
    tools=tools,
    tool_choice="auto",
)

response


# In[ ]:


tool_function = response.choices[0].message.tool_calls[0].function
tool_function


# In[29]:


tool_function.name


# In[30]:


import json

args = json.loads(tool_function.arguments)
args


# In[31]:


function_result = names_to_functions[tool_function.name](**args)
function_result


# ## More about RAG
# To learn about more advanced chunking and retrieval methods, you can check out:
# - [Advanced Retrieval for AI with Chroma](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/1/introduction)
#   - Sentence window retrieval
#   - Auto-merge retrieval
# - [Building and Evaluating Advanced RAG Applications](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag)
#   - Query Expansion
#   - Cross-encoder reranking
#   - Training and utilizing Embedding Adapters
# 

# In[ ]:




