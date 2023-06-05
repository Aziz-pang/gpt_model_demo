import gradio as gr
import sys
import logging
import sys
import os

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = os.getenv('API_CHAT_GPT_CHOO')


def construct_index(directory_path):
    context_window = 2058
    max_input_size = 4096
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

    # documents = SimpleDirectoryReader(directory_path).load_data()
    # llm_predictor = LLMPredictor(llm=ChatOpenAI(
    #     temperature=0, model_name="gpt-3.5-turbo", streaming=True))
    # service_context = ServiceContext.from_defaults(
    #     llm_predictor=llm_predictor, chunk_size=512)
    # index = GPTVectorStoreIndex.from_documents(
    #     documents, service_context=service_context)

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    print(index)
    return index


def chatbot(input_text):
    index = GPTVectorStoreIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(
                         lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")
index = construct_index("testData")
iface.launch(share=True)
