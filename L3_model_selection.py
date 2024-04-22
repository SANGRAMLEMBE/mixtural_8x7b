#!/usr/bin/env python
# coding: utf-8

# # Model Selection

# ### Get API Key

# In[ ]:


from helper import load_mistral_api_key
api_key, dlai_endpoint = load_mistral_api_key(ret_key=True)


# - Note: in the classroom, if you print out this `api_key` variable, it is not a real API key (for security reasons).
# - If you wish to run this code on your own machine, outside of the classroom, you can still reuse the code that you see in `helper.py`.
# - It uses [python-dotenv](https://pypi.org/project/python-dotenv/) library to securely save and load sensitive information such as API keys.

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


# ## Mistral Small
# 
# Good for simple tasks, fast inference, lower cost.
# - classification

# In[ ]:


prompt = """
Classify the following email to determine if it is spam or not.
Only respond with the exact text "Spam" or "Not Spam". 

# Email:
🎉 Urgent! You've Won a $1,000,000 Cash Prize! 
💰 To claim your prize, please click on the link below: 
https://bit.ly/claim-your-prize
"""


# In[ ]:


mistral(prompt, model="mistral-small-latest")


# ## Mistral Medium
# 
# Good for intermediate tasks such as language transformation.
# - Composing text based on provided context (e.g. writing a customer service email based on purchase information).

# In[ ]:


prompt = """
Compose a welcome email for new customers who have just made 
their first purchase with your product. 
Start by expressing your gratitude for their business, 
and then convey your excitement for having them as a customer. 
Include relevant details about their recent order. 
Sign the email with "The Fun Shop Team".

Order details:
- Customer name: Anna
- Product: hat 
- Estimate date of delivery: Feb. 25, 2024
- Return policy: 30 days
"""


# In[ ]:


response_medium = mistral(prompt, model="mistral-medium-latest")


# In[ ]:


print(response_medium)


# ## Mistral Large: 
# 
# Good for complex tasks that require advanced reasoning.
# - Math and reasoning with numbers.

# In[ ]:


prompt = """
Calculate the difference in payment dates between the two \
customers whose payment amounts are closest to each other \
in the following dataset. Do not write code.

# dataset: 
'{
  "transaction_id":{"0":"T1001","1":"T1002","2":"T1003","3":"T1004","4":"T1005"},
    "customer_id":{"0":"C001","1":"C002","2":"C003","3":"C002","4":"C001"},
    "payment_amount":{"0":125.5,"1":89.99,"2":120.0,"3":54.3,"4":210.2},
"payment_date":{"0":"2021-10-05","1":"2021-10-06","2":"2021-10-07","3":"2021-10-05","4":"2021-10-08"},
    "payment_status":{"0":"Paid","1":"Unpaid","2":"Paid","3":"Paid","4":"Pending"}
}'
"""


# In[ ]:


response_small = mistral(prompt, model="mistral-small-latest")


# In[ ]:


print(response_small)


# In[ ]:


response_large = mistral(prompt, model="mistral-large-latest")


# In[ ]:


print(response_large)


# ## Expense reporting task

# In[ ]:


transactions = """
McDonald's: 8.40
Safeway: 10.30
Carrefour: 15.00
Toys R Us: 20.50
Panda Express: 10.20
Beanie Baby Outlet: 25.60
World Food Wraps: 22.70
Stuffed Animals Shop: 45.10
Sanrio Store: 85.70
"""

prompt = f"""
Given the purchase details, how much did I spend on each category:
1) restaurants
2) groceries
3) stuffed animals and props
{transactions}
"""


# In[ ]:


response_small = mistral(prompt, model="mistral-small-latest")
print(response_small)


# In[ ]:


response_large = mistral(prompt, model="mistral-large-latest")
print(response_large)


# ## Writing and checking code

# In[ ]:


user_message = """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Your code should pass these tests:

assert twoSum([2,7,11,15], 9) == [0,1]
assert twoSum([3,2,4], 6) == [1,2]
assert twoSum([3,3], 6) == [0,1]
"""


# In[ ]:


print(mistral(user_message, model="mistral-large-latest"))


# ### Try out the code that the model provided
# - Copy the code that the model provided and try running it!
# 
# Here is the code that was output at the time of filming:
# ```Python
# def twoSum(nums, target):
#     seen = {}
#     for i, num in enumerate(nums):
#         complement = target - num
#         if complement in seen:
#             return [seen[complement], i]
#         seen[num] = i
# ```
# - Also try running the assert statements in the original prompt
# ```Python
# assert twoSum([2,7,11,15], 9) == [0,1]
# assert twoSum([3,2,4], 6) == [1,2]
# assert twoSum([3,3], 6) == [0,1]
# ```

# In[ ]:





# In[ ]:





# ## Natively Fluent in English, French, Spanish, German, and Italian
# - This means that you can use Mistral models for more than translating from one language to another.
# - If you are a native Spanish speaker, for instance, you can communicate with Mistral models in Spanish for any of your tasks.

# In[ ]:


user_message = """
Lequel est le plus lourd une livre de fer ou un kilogramme de plume
"""


# In[ ]:


print(mistral(user_message, model="mistral-large-latest"))


# ### Try it out for yourself
# - Try communicating with the Mistral Large model in Spanish
#   - (If you need help, you can first translate a prompt from English to Spanish, and then prompt the model in Spanish).

# ## List of Mistral models that you can call:
# 
# You can also call the two open source mistral models via API calls.
# Here is the list of models that you can try:
# ```
# open-mistral-7b
# open-mixtral-8x7b
# mistral-small-latest
# mistral-medium-latest
# mistral-large-latest
# ```
# 
# For example:
# ```Python
# mistral(prompt, model="open-mixtral-8x7b")
# ```

# In[ ]:




