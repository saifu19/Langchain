# Contains a working copy of the ProjectWe Brain that is connected to the ProjectWe pinecone and a testing client pinecone
# Run the code

# Imports
import os
import warnings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Pinecone imports
import pinecone
from langchain.vectorstores import Pinecone

# OpenAI imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Chain imports
from langchain.chains.router import MultiRetrievalQAChain

# Agent imports
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# Memory imports
from langchain.memory.buffer import ConversationBufferMemory

# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
	environment='us-west4-gcp'      
)
index_name = "mojosolo-main"

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    temperature = 0.1,
    model_name="gpt-4"
)

# Initialize retrievers for MultiRetrievalQA Chain
client_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-projectwe-client-pinecone").as_retriever()
projectwe_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-projectwe-mojomosaic-pinecone").as_retriever()
muse_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-muse-mojomosaic-pinecone").as_retriever()
retriever_infos = [
    {
        "name": "client retriever", 
        "description": "Good for answering questions about people", 
        "retriever": client_retriever
    },
    {
        "name": "projectwe retriever", 
        "description": "Good for answering miscellaneous questions",
        "retriever": projectwe_retriever
    },
    {
        "name": "muse retriever", 
        "description": "Good for answering miscellaneous questions",
        "retriever": muse_retriever
    }
]

# Initialize Multiretrieval QA Chain and add to tools
qaChain = MultiRetrievalQAChain.from_retrievers(
    llm=llm, 
    retriever_infos=retriever_infos)  # Add verbose = True to see inner workings

# Custom tool function for upserting to pinecone
def upsertToPinecone(mem):
    Pinecone.from_texts(texts=[mem], index_name=index_name, embedding=embeddings, namespace="cust-projectwe-client-pinecone")
    return "Saved " + mem + " to client database"

tools=[
    Tool.from_function(
        func=qaChain.run,
        name="Search pinecone",
        description="Useful for when you need to answer questions"
    ),
    Tool.from_function(
        func=lambda mem: upsertToPinecone(mem),
        name="Save to user database",
        description="Useful for when you need to save information to the user's database"
    )
]

# Memory (currently a Conversation Buffer Memory, will become Motorhead)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up Agent
agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory) # Add verbose = True to see inner workings
print("Enter your first query: ")
prompt = input()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while(prompt.lower() != "quit"):
        print("MojoBob: ")
        print(agent_executor.run(prompt))
        print("Human: ")
        prompt = input()