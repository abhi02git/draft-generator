import os
from apikey import apikey

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import streamlit as st

#setting up the openai api
os.environ['OPENAI_API_KEY'] = apikey


#App framework
st.title('Draft maker')
prompt = st.text_input('Plug in your prompt')


#creating the list
list_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'You are a lawyer. I want you to make a contract draft on {topic} draft. Give me the bullet list of informations needed for making the draft within 30 words'
)

#creating the document
document_template = PromptTemplate(
    input_variables = ['list1'],
    template = 'Use ChatGPT to make a draft form containing these :  {list1} along with their signatures'
)

#LLMs and chains
llm = OpenAI(temperature = 0.9)

list_chain = LLMChain(llm = llm, prompt = list_template, verbose = True)
document_chain = LLMChain(llm = llm, prompt = document_template, verbose = True)
sequential_chain = SimpleSequentialChain(chains = [list_chain, document_chain], verbose = True)

#output
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)