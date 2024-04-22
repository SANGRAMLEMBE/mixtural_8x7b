#!/usr/bin/env python
# coding: utf-8

# # Function Calling
# 

# In[ ]:


# !pip install pandas "mistralai>=0.1.2"


# ### Load API key

# In[ ]:


from helper import load_mistral_api_key
api_key, dlai_endpoint = load_mistral_api_key(ret_key=True)


# In[ ]:


import pandas as pd


# In[ ]:


data = {
    "transaction_id": ["T1001", "T1002", "T1003", "T1004", "T1005"],
    "customer_id": ["C001", "C002", "C003", "C002", "C001"],
    "payment_amount": [125.50, 89.99, 120.00, 54.30, 210.20],
    "payment_date": [
        "2021-10-05",
        "2021-10-06",
        "2021-10-07",
        "2021-10-05",
        "2021-10-08",
    ],
    "payment_status": ["Paid", "Unpaid", "Paid", "Paid", "Pending"],
}
df = pd.DataFrame(data)


# In[ ]:


df


# #### How you might answer data questions without function calling
# - Not recommended, but an example to serve as a contrast to function calling.

# In[ ]:


data = """
    "transaction_id": ["T1001", "T1002", "T1003", "T1004", "T1005"],
    "customer_id": ["C001", "C002", "C003", "C002", "C001"],
    "payment_amount": [125.50, 89.99, 120.00, 54.30, 210.20],
    "payment_date": [
        "2021-10-05",
        "2021-10-06",
        "2021-10-07",
        "2021-10-05",
        "2021-10-08",
    ],
    "payment_status": ["Paid", "Unpaid", "Paid", "Paid", "Pending"],
}
"""
transaction_id = "T1001"

prompt = f"""
Given the following data, what is the payment status for \
 transaction_id={transaction_id}?

data:
{data}

"""


# In[ ]:


import os
from mistralai.client import MistralClient
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


# In[ ]:


response = mistral(prompt)
print(response)


# ## Step 1. User: specify tools and query
# 
# ### Tools
# 
# - You can define all tools that you might want the model to call.

# In[ ]:


import json


# In[ ]:


def retrieve_payment_status(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps(
            {"status": df[df.transaction_id == transaction_id].payment_status.item()}
        )
    return json.dumps({"error": "transaction id not found."})


# In[ ]:


status = retrieve_payment_status(df, transaction_id="T1001")
print(status)


# In[ ]:


type(status)


# In[ ]:


def retrieve_payment_date(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps(
            {"date": df[df.transaction_id == transaction_id].payment_date.item()}
        )
    return json.dumps({"error": "transaction id not found."})


# In[ ]:


date = retrieve_payment_date(df, transaction_id="T1002")
print(date)


# - You can outline the function specifications with a JSON schema.

# In[ ]:


tool_payment_status = {
    "type": "function",
    "function": {
        "name": "retrieve_payment_status",
        "description": "Get payment status of a transaction",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction id.",
                }
            },
            "required": ["transaction_id"],
        },
    },
}


# In[ ]:


type(tool_payment_status)


# In[ ]:


tool_payment_date = {
    "type": "function",
    "function": {
        "name": "retrieve_payment_date",
        "description": "Get payment date of a transaction",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction id.",
                }
            },
            "required": ["transaction_id"],
        },
    },
}


# In[ ]:


type(tool_payment_status)


# In[ ]:


tools = [tool_payment_status, tool_payment_date]


# In[ ]:


type(tools)


# In[ ]:


tools


# ### functools

# In[ ]:


import functools


# In[ ]:


names_to_functions = {
    "retrieve_payment_status": functools.partial(retrieve_payment_status, df=df),
    "retrieve_payment_date": functools.partial(retrieve_payment_date, df=df),
}


# In[ ]:


names_to_functions["retrieve_payment_status"](transaction_id="T1001")


# In[ ]:


tools


# ### User query
# 
# - Example: “What’s the status of my transaction?”

# In[ ]:


from mistralai.models.chat_completion import ChatMessage

chat_history = [
    ChatMessage(role="user", content="What's the status of my transaction?")
]


# ## Step 2. Model: Generate function arguments 

# In[ ]:


from mistralai.client import MistralClient

model = "mistral-large-latest"

client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"), endpoint=os.getenv("DLAI_MISTRAL_API_ENDPOINT"))

response = client.chat(
    model=model, messages=chat_history, tools=tools, tool_choice="auto"
)

response


# In[ ]:


response.choices[0].message.content


# ### Save the chat history

# In[ ]:


chat_history.append(
    ChatMessage(role="assistant", content=response.choices[0].message.content)
)
chat_history.append(ChatMessage(role="user", content="My transaction ID is T1001."))
chat_history


# In[ ]:


response = client.chat(
    model=model, messages=chat_history, tools=tools, tool_choice="auto"
)


# In[ ]:


response


# In[ ]:


response.choices[0].message


# In[ ]:


chat_history.append(response.choices[0].message)


# - Notice these fields:
# - `name='retrieve_payment_status'`
# - `arguments='{"transaction_id": "T1001"}'`

# ## Step 3. User: Execute function to obtain tool results
# 
# - Currently, the user is the one who will execute these functions (the model will not execute these functions on its own).

# In[ ]:


tool_function = response.choices[0].message.tool_calls[0].function
print(tool_function)


# In[ ]:


tool_function.name


# In[ ]:


tool_function.arguments


# - The function arguments are expected to be in a Python dictionary and not a string.
# - To make this string into a dictionary, you can use `json.loads()`  

# In[ ]:


args = json.loads(tool_function.arguments)
print(args)


# - Recall the functools dictionary that you made earlier
# 
# ```Python
# import functools
# names_to_functions = {
#     "retrieve_payment_status": 
#       functools.partial(retrieve_payment_status, df=df),
#     
#     "retrieve_payment_date": 
#       functools.partial(retrieve_payment_date, df=df),
# }
# ```

# In[ ]:


function_result = names_to_functions[tool_function.name](**args)
function_result


# - The output of the function call can be saved as a chat message, with the role "tool".

# In[ ]:


tool_msg = ChatMessage(role="tool", name=tool_function.name, content=function_result)
chat_history.append(tool_msg)


# In[ ]:


chat_history


# ## Step 4. Model: Generate final answer
# - The model can now reply to the user query, given the information provided by the "tool".

# In[ ]:


response = client.chat(model=model, messages=chat_history)
response.choices[0].message.content


# ### Try it for yourself!
# - Try asking another question about the data, such as "how much did I pay my recent order?
#   - You can be customer "T1002", for instance

# In[ ]:




