# Import the required modules
from langchain_google_genai import GoogleGenerativeAI  # Use to access the Google Generative AI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

# the icon and the title of the web app
st.set_page_config(page_title="Country Information", page_icon="🌐")

# streamlit framework
st.title('Country Information')
input_country = st.text_input("Enter a country:")

# Google Generative AI
google_api_key = os.getenv("GOOGLE_GEMINI_AI")  # Google API Key
google_llm = GoogleGenerativeAI(temperature=0.8, google_api_key=google_api_key, model="gemini-pro")

# Prompt Templates
country_info_prompt = PromptTemplate(
    input_variables=['country'],
    template="Tell me about {country}"
)

capital_prompt = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)

population_prompt = PromptTemplate(
    input_variables=['country'],
    template="What is the population of {country}?"
)

# Chains of LLMs
country_info_chain = LLMChain(
    llm=google_llm,
    prompt=country_info_prompt,
    verbose=True,
    output_key='country_info'
)

capital_chain = LLMChain(
    llm=google_llm,
    prompt=capital_prompt,
    verbose=True,
    output_key='capital'
)

population_chain = LLMChain(
    llm=google_llm,
    prompt=population_prompt,
    verbose=True,
    output_key='population'
)

# Parent Chain
country_info_chain = SequentialChain(
    chains=[country_info_chain, capital_chain, population_chain],
    input_variables=['country'],
    output_variables=['country_info', 'capital', 'population'],
    verbose=True
)

# Fetch information
if input_country:
    country_result = country_info_chain({'country': input_country})

    with st.expander(f"Information about {input_country}"):
        st.write(country_result['country_info'])

    with st.expander(f"Capital of {input_country}"):
        st.write(country_result['capital'])

    with st.expander(f"Population of {input_country}"):
        st.write(country_result['population'])
