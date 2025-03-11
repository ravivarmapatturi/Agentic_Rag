import os
import yaml 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import streamlit as st

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional
from pydantic import BaseModel, PrivateAttr
from langchain_core.runnables import RunnableSerializable
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# with open("/home/ravivarma/Downloads/preplaced/session_5_tasks/miniproject/credentials/api_keys.yaml") as file:
#     config = yaml.safe_load(file)
# api_keys = config['api_keys']["chatgpt"]
# api_groq = config["api_keys"]["groq"]
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"]
# from huggingface_hub import InferenceClient
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["GROQ_API_KEY"] = st.secrets["groq"]


# chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)




model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)



llm_model="mistral"

if llm_model=="gpt-3.5-turbo" or llm_model=="gpt-4":
    chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)


if llm_model=="mistral":
    chatgpt= ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )




# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
#     ("system", "Context: {context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])

# question_answer_chain = create_stuff_documents_chain(chatgpt, qa_prompt)












# with open("/home/ravivarma/Downloads/preplaced/session_5_tasks/miniproject/credentials/api_keys.yaml") as file:
#     config = yaml.safe_load(file)
# api_keys = config['api_keys']["chatgpt"]
# os.environ["OPENAI_API_KEY"] = api_keys

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)


qa_template1 = """
You are an intelligent and detail-oriented QA Chatbot designed to generate comprehensive and accurate answers. 
Your primary task is to provide a clear, detailed, and context-aware response to the user's question.

Instructions:
- Carefully analyze the provided context and use it to craft a complete and precise answer.
- If the question is ambiguous or lacks sufficient detail, take help from user .
- Ensure the answer is structured, easy to understand, and includes examples or explanations where necessary.
=======
- use provided context whenever it is necessary ,otherwise dont use it .

Context:
{context}

Question: {question}

Answer:
"""

prompt= ChatPromptTemplate.from_template(qa_template1)
rag_chain = LLMChain(prompt=prompt, llm=chatgpt)

