import os
import openai
openai.organization = "org-H3FDdYKWRd5SOFkz4hPh0lTe"
# openai.api_key = os.getenv('API_CHAT_GPT')
openai.api_key = os.getenv('API_AZURE_OPENAI')
list = openai.Model.list()
print(list)
