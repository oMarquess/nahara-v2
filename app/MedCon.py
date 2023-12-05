import os
import openai
import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Load .env file from the project's root directory
load_dotenv()

llm_name = "gpt-3.5-turbo-1106"
persist_directory = 'docs/chroma/'
# Retrieve the API key from the .env file
# The key in the .env file is 'OPENAI_API_KEY'
openai.api_key = os.getenv('OPENAI_API_KEY')

# print(openai.api_key)

# Use the recommended model
# current_date = datetime.datetime.now().date()
# if current_date < datetime.date(2023, 9, 2):
#     llm_name = "gpt-3.5-turbo-0301"
# else:
#     llm_name = "gpt-3.5-turbo-1106"
# print(llm_name)

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)