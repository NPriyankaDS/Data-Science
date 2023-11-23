import streamlit as st
from langchain.llms import GooglePalm
import os
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

st.title("Know about your celebrity")
st.header("Let's get started!!")

input_text = st.text_input("Enter the name of the celebrity")

#prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)   

chain = LLMChain(llm = llm,prompt=first_input_prompt,verbose= True,output_key='person')

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was the {person} born?"
)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Major five events occurred around {dob} in the world"
)

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='events')

parent_chain = SequentialChain(
    chains=[chain,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','events'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))

