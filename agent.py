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
from langchain.chains import RetrievalQA

# Agent imports
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool

# Memory imports
from langchain.memory.buffer import ConversationBufferMemory

# Flask imports
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
    environment='us-west4-gcp'      
)
# Used for Pinecone.from_existing_index
index_name = "mojosolo-main"
# Used for retrieving namespaces
pineconeIndex = pinecone.Index('mojosolo-main')

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    temperature = 0.1,
    model_name="gpt-4"
)

# Initializes the tool lists
tools=[]
toolDescriptions = []

# Memory (currently a Conversation Buffer Memory, will become Motorhead)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up Agent
agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory) # Add verbose = True to see inner workings

# Custom tool for retrieving
def retrievalTool(namespace, name=None, description=None, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    # If they didn't set name/description, use defaults
    if(name == None or name == "Use Default"):
        name="Retrieve from " + namespace + " database"
    if(description == None or description == "Use Default"):
        description = "Useful for when you need to retrieve information from the " + namespace + " database"

    toolDescriptions.insert(id, {"name": name, "description": description, "namespace": namespace, "type": "Retrieval"})

    retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace).as_retriever(search_kwargs={'k': 10})
    
    newTool = create_retriever_tool(
        retriever,
        name,
        description,
    )

    tools.insert(id, newTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory)

# Custom tool for upserting; uses the helper function "upsertToPinecone"
def upsertTool(namespace, name=None, description=None, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    # If they didn't set name/description, use defaults
    if(name == None or name == "Use Default"):
        name="Save to " + namespace + " database"
    if(description == None or description == "Use Default"):
        description = "Useful for when you need to save information to the " + namespace + " database"

    toolDescriptions.insert(id, {"name": name, "description": description, "namespace": namespace, "type": "Upsert"})

    newTool = Tool.from_function(
        func=lambda mem: upsertToPinecone(mem, namespace),
        name=name,
        description=description
    )

    tools.insert(id, newTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory)


# Helper function for upserting to pinecone
def upsertToPinecone(mem, namespace):
    Pinecone.from_texts(texts=[mem], index_name=index_name, embedding=embeddings, namespace=namespace)
    return "Saved " + mem + " to " + namespace + " database"

# Initializes the base tools the agent should start with
# retrievalTool("cust-projectwe-client-pinecone")
# upsertTool("cust-projectwe-client-pinecone")

# Runs Bob
@app.route('/', methods=['POST'])
def mojoBob():
    # If the user submits a query
    if(request.form.get("inp")):
        # Run the agent_executor and display the agent's response (with chat history)
        prompt = request.form.get("inp")
        agent_executor.run(prompt)
        
        return redirect('/')

# Helper function that actually does the deletion of a tool
def deleteToolHelper(index):
    tools.pop(index)
    toolDescriptions.pop(index)

# Loads Homepage
@app.route('/', methods=['GET'])
def viewIndex():
    mem=memory.load_memory_variables({})["chat_history"]
    return render_template('index.html', memory=mem, size=len(mem))

# Loads Namespace CRUD
@app.route('/namespace-crud', methods=['GET'])
def namespaceViewIndex():
    return render_template('namespaces/crud.html', namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()))

# Loads Namespace Create View
@app.route('/namespace-create', methods=['GET'])
def namespaceViewCreate():
    return render_template('namespaces/create.html', namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()))

# Loads Namespace Edit View
@app.route('/namespace-edit/<name>', methods=['GET'])
def namespaceViewEdit(name):
    return render_template('namespaces/edit.html', namespace=name)

# Loads Tool CRUD
@app.route('/tool-crud', methods=['GET'])
def toolViewIndex():
    agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory, verbose = True)
    return render_template('tools/crud.html', tools=toolDescriptions)

# Loads Tool Create View
@app.route('/tool-create', methods=['GET'])
def toolViewCreate():
    return render_template('tools/create.html', namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()))

# Loads Tool Edit View
@app.route('/tool-edit/<id>', methods=['GET'])
def toolViewEdit(id):
    tool = toolDescriptions[int(id) - 1]
    return render_template('tools/edit.html', id=id, tool=tool, namespaces=sorted(pineconeIndex.describe_index_stats()['namespaces'].items()))

# Creates Namespace
@app.route('/namespace-create', methods=['POST'])
def createNamespace():
    namespace = request.form.get("namespace")
    # upserts the namespace to pinecone
    upsertToPinecone(request.form.get("text"), namespace)

    # If they wish to create tools for the given namespace
    if(request.form.get("tools")):
        # Add an upsert tool
        upsertTool(namespace)

        # Add a retrieval tool
        retrievalTool(namespace)

    return redirect(url_for('namespaceViewIndex'))

# Edits Namespace (Upsert Data)
@app.route('/namespace-edit', methods=['POST'])
def editNamespace():
    upsertToPinecone(request.form.get("text"), request.form.get("namespace"))
    return redirect(url_for('namespaceViewIndex'))

# Deletes Namespace
@app.route('/namespace-crud', methods=['POST'])
def deleteNamespace():
    namespace = request.form.get("namespace")

    # deletes the namespace on pinecone
    pineconeIndex.delete(delete_all=True, namespace=namespace)

    # deletes associated tools
    for idx, tool in enumerate(toolDescriptions):
        if tool["namespace"] == namespace:
            deleteToolHelper(idx)

    return redirect(url_for('namespaceViewIndex'))

# Creates Tool
@app.route('/tool-create', methods=['POST'])
def createTool():
    namespace = request.form.get("namespace")
    type = request.form.get("type")

    # If it is an upsert tool
    if(type == "Upsert"):
        upsertTool(namespace, request.form.get("name"), request.form.get("description"))
    # Else if it is a retrieval tool
    elif(type == "Retrieval"):
        retrievalTool(namespace, request.form.get("name"), request.form.get("description"))
    
    return redirect(url_for('toolViewIndex'))

# Edits Tool
@app.route('/tool-edit', methods=['POST'])
def editTool():
    id = int(request.form.get("id")) - 1

    # Deletes the old tool
    deleteToolHelper(id)

    # Inserts the new tools into the old location
    if(request.form.get("type") == "Upsert"):
        upsertTool(request.form.get("namespace"), request.form.get("name"), request.form.get("description"), id)
    elif (request.form.get("type") == "Retrieval"):
        retrievalTool(request.form.get("namespace"), request.form.get("name"), request.form.get("description"), id)
    

    return redirect(url_for('toolViewIndex'))

# Deletes Tool
@app.route('/tool-crud', methods=['POST'])
def deleteTool():
    # Deletes both the tool description and the tool itself
    deleteToolHelper(int(request.form.get("tool")) - 1)
    return redirect(url_for('toolViewIndex'))


# Custom tool creator
def customTool(function, name, description, id=len(tools)+1):
    global agent_executor, tools, llm, memory

    toolDescriptions.insert(id, {"name": name, "description": description, "namespace": "None", "type": "Upsert"})

    newTool = Tool.from_function(
        func=lambda input: function(input),
        name=name,
        description=description
    )

    tools.insert(id, newTool)
    agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory)