import gradio as gr
import sys
import logging
import sys
import os

from llama_index import SimpleDirectoryReader, load_index_from_storage, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext
from langchain import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = os.getenv('API_CHAT_GPT_CHOO')


def construct_index(directory_path):
    context_window = 2058
    num_outputs = 512
    max_chunk_overlap = 20
    tokenizer = None
    separator = ''
    embedding_limit = None
    chunk_overlap_ratio = 0.2
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        context_window,  num_outputs, chunk_overlap_ratio, chunk_size_limit, tokenizer, separator, embedding_limit, max_chunk_overlap)
    # 模型 gpt-3.5-turbo
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.7, model_name="gpt-3.5-turbo3", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist('persist_dir')
    return index

from llama_index.query_engine import RetrieverQueryEngine

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="persist_dir")
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever()
    query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='compact')
    response = query_engine.query(input_text)
    return str(response)


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(
                         lines=7, label="请输入你的问题"),
                     outputs=gr.Textbox(lines=7, label="机器人的回答"),
                     title="AI Chatbot 训练")
index = construct_index("testData")
iface.launch(share=True)
