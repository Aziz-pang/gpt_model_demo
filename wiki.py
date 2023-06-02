import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# fetch "New York City" page from Wikipedia
from pathlib import Path

import requests
response = requests.get(
    'https://en.wikipedia.org/w/api.php',
    params={
        'action': 'query',
        'format': 'json',
        'titles': 'New York City',
        'prop': 'extracts',
        # 'exintro': True,
        'explaintext': True,
    }
).json()
page = next(iter(response['query']['pages'].values()))
nyc_text = page['extract']

data_path = Path('data')
if not data_path.exists():
    Path.mkdir(data_path)

with open('data/nyc_text.txt', 'w', encoding='utf-8') as fp:
    fp.write(nyc_text)
# My OpenAI Key
import os
os.environ["OPENAI_API_KEY"] = os.getenv('API_CHAT_GPT')

from llama_index import GPTTreeIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTTreeIndex.from_documents(documents)
# GPT doesn't find the corresponding evidence in the leaf node, but still gives the correct answer
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
query_engine.query("What is the name of the professional women's basketball team in New York City?")
# GPT doesn't find the corresponding evidence in the leaf node, but still gives the correct answer
# set Logging to DEBUG for more detailed outputs

query_engine.query("What battles took place in New York City in the American Revolution?")
# GPT doesn't find the corresponding evidence in the leaf node, but still gives the correct answer
# set Logging to DEBUG for more detailed outputs

query_engine.query("What are the airports in New York City?")
# Try using embedding query
query_engine.query("What are the airports in New York City?", retriever_mode="embedding")