#!/usr/bin/env python
# coding: utf-8

# # Chat Interface

# In[1]:


from helper import load_mistral_api_key
api_key, dlai_endpoint = load_mistral_api_key(ret_key=True)


# In[2]:


import os
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient


# ## Panel
# 
# [Panel](https://panel.holoviz.org/) is an open source python library that you can use to create dashboards and apps.

# In[3]:


import panel as pn
pn.extension()


# ## Basic Chat UI

# In[4]:


def run_mistral(contents, user, chat_interface):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    messages = [ChatMessage(role="user", content=contents)]
    chat_response = client.chat(
        model="mistral-large-latest", 
        messages=messages)
    return chat_response.choices[0].message.content


# In[5]:


chat_interface = pn.chat.ChatInterface(
    callback=run_mistral, 
    callback_user="Mistral"
)

chat_interface


# In[ ]:





# ## RAG UI
# 
# Below is the RAG code that you used in the previous lesson.

# In[6]:


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

# Optionally save this text into a file.
file_name = "AI_greenhouse_gas.txt"
with open(file_name, 'w') as file:
    file.write(text)


# In[7]:


import numpy as np
import faiss

client = MistralClient(
    api_key=os.getenv("MISTRAL_API_KEY"),
    endpoint=os.getenv("DLAI_MISTRAL_API_ENDPOINT")
)

prompt = """
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(model="mistral-embed", input=input)
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [ChatMessage(role="user", content=user_message)]
    chat_response = client.chat(model=model, messages=messages)
    return chat_response.choices[0].message.content

def answer_question(question, user, instance):
    text = file_input.value.decode("utf-8")

    # split document into chunks
    chunk_size = 2048
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
    # generate response based on the retrieved relevant text chunks
    response = run_mistral(
        prompt.format(retrieved_chunk=retrieved_chunk, question=question)
    )
    return response


# ### Connect the Chat interface with your user-defined function
# 
# - Note, you can find some sample text files to upload to this RAG UI by clicking on the 'Jupyter' logo and to view the file directory of the lesson.
# - Or you can create any text file and copy-paste some text from a web article.

# In[8]:


file_input = pn.widgets.FileInput(accept=".txt", value="", height=50)

chat_interface = pn.chat.ChatInterface(
    callback=answer_question,
    callback_user="Mistral",
    header=pn.Row(file_input, "### Upload a text file to chat with it!"),
)
chat_interface.send(
    "Send a message to get a reply from Mistral!", 
    user="System", 
    respond=False
)
chat_interface


# #### Note about Panel
# If you are running this notebook on your local machine:
# - After uploading a local text file, and when typing into the textbox for the RAG UI, you may notice that the screen "jumps" downward, and a new code cell is inserted above the chat UI.
# - The letter "a" is a jupyter notebook shortcut that inserts a new code cell "above" the current one.
# - If you see this, please use your mouse cursor to click back into the "Send a message" textbox and continue typing.  After that, you will not see the screen jump even if you type a letter "a" or any other jupyter notebook shortcut key.
# 
# For a more permanent fix, please upgrade to the latest version of juypter lab and panel.

# In[ ]:





# ### Try it for yourself!
# - Try writing some other user-defined function that can then be connected to a Panel app.
#   - For instance, you can take the function calling example from earlier in the course and add a chat interface to it.

# ## Did you like this course?
# 
# - If you liked this course, could you consider giving a rating and share what you liked? 💕
# - If you did not like this course, could you also please share what you think could have made it better? 🙏
# 
# #### A note about the "Course Review" page.
# The rating options are from 0 to 10.
# - A score of 9 or 10 means you like the course.😻 💕
# - A score of 7 or 8 means you feel neutral about the course (neither like nor dislike).😼🙄
# - A score of 0,1,2,3,4,5 or 6 all mean that you do not like the course. 😿😭 
#   - Whether you give a 0 or a 6, these are all defined as "detractors" according to the standard measurement called "Net Promoter Score". 🧐

# In[ ]:




