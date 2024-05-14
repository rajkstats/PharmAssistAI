# chainlit run app.py -w
# Standard library imports
import asyncio
import io
import json
import os
import re
import requests
import zipfile

# Data handling
import pandas as pd

# Environment variables
from dotenv import load_dotenv

# Typing for function signatures
from typing import Any, List, Optional

# Bioinformatics
from Bio import Entrez, Medline

# ChainLit specific imports
import chainlit as cl
from chainlit.types import AskFileResponse

# Langchain imports for AI and chat models
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.evaluation import StringEvaluator
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler
from langchain_openai import OpenAI, OpenAIEmbeddings

# Vector storage and document loading
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient

# Custom evaluations
from custom_eval import PharmAssistEvaluator, HarmfulnessEvaluator, AIDetectionEvaluator

# LangSmith for client interaction
from langsmith import Client


langsmith_client = Client()

# Load environment variables from a .env file
load_dotenv()

# Define system template for the chatbot
system_template = """
You are , an AI assistant for pharmacists and pharmacy students. Use the following pieces of context to answer the user's question.

If you don't know the answer, simply state that you don't have enough information to provide an answer. Do not attempt to make up an answer.

ALWAYS include a "SOURCES" section at the end of your response, referencing the specific documents from which you derived your answer. 

If the user greets you with a greeting like "Hi", "Hello", or "How are you", respond in a friendly manner.

Example response format:
<answer>
SOURCES: <document_references>

Begin!
----------------
{summaries}
"""

# Define messages for the chatbot prompt
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

qdrant_vectorstore = None

# Function to search for related papers on PubMed
async def search_related_papers(query, max_results=3):
    """
    Search PubMed for papers related to the provided query and return a list of formatted strings with paper details and URLs.
    """
    try:
        # Set up Entrez email (replace with your email)
        Entrez.email = os.environ.get("ENTREZ_EMAIL")

        # Search PubMed for related papers
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        # Retrieve the details of the related papers
        id_list = record["IdList"]
        if not id_list:
            return ["No directly related papers found. Try broadening your search query."]

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle)

        related_papers = []
        for record in records:
            title = record.get("TI", "")
            authors = ", ".join(record.get("AU", []))
            citation = f"{authors}. {title}. {record.get('SO', '')}"
            url = f"https://pubmed.ncbi.nlm.nih.gov/{record['PMID']}/"
            related_papers.append(f"[{citation}]({url})")

        if not related_papers:
            related_papers = ["No directly related papers found. Try broadening your search query."]

        return related_papers
    except Exception as e:
        print(f"Error occurred while searching for related papers: {e}")
        return ["An error occurred while searching for related papers. Please try again later."]

# Function to generate related questions based on retrieved results
async def generate_related_questions(retrieved_results, num_questions=2, max_tokens=50):
    """
    Generate related questions based on the provided retrieved results from a document store.
    """
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["context"],
        template="Given the following context, generate {num_questions} related questions:\n\nContext: {context}\n\nQuestions:",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    context = " ".join([doc.page_content for doc in retrieved_results])
    generated_questions = chain.run(context=context, num_questions=num_questions, max_tokens=max_tokens)

    # Remove numbering from the generated questions
    related_questions = [question.split(". ", 1)[-1] for question in generated_questions.split("\n") if question.strip()]

    return related_questions

# Function to generate answer based on user's query
async def generate_answer(query):
    """
    Generate an answer to the user's query using a conversational retrieval chain and handle callbacks for related questions and papers.
    """
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=qdrant_vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    try:
        cb = cl.AsyncLangchainCallbackHandler()
        #evaluator = PharmAssistEvaluator()
        feedback_callback = EvaluatorCallbackHandler(evaluators=[PharmAssistEvaluator(),HarmfulnessEvaluator(),AIDetectionEvaluator()])
        res = await chain.acall(query, callbacks=[cb,feedback_callback])
        answer = res["answer"]
        source_documents = res["source_documents"]

        if answer.lower().startswith("i don't know") or answer.lower().startswith("i don't have enough information"):
            return answer, [], [], [],[]

        text_elements = []
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\n\n**SOURCES:** {', '.join(source_names)}"
            else:
                answer += "\n\n**SOURCES:** No sources found"

        related_questions = await generate_related_questions(source_documents)
        related_question_actions = [
            cl.Action(name="related_question", value=question.strip(), label=question.strip())
            for question in related_questions if question.strip()
        ]

        # Search for related papers on PubMed
        related_papers = await search_related_papers(query)

        return answer, text_elements, related_question_actions, related_papers, query

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while processing your request. Please try again later.", [], [], [],[], query

# Action callback for related question selection
@cl.action_callback("related_question")
async def on_related_question_selected(action: cl.Action):
    """
    Handle the selection of a related question, generate and send answers and further interactions.
    """
    question = action.value
    await cl.Message(content=question, author="User").send()

    answer, text_elements, related_question_actions, related_papers, query = await generate_answer(question)
    await cl.Message(content=answer, elements=text_elements, author="PharmAssistAI").send()

    # Send related questions as a separate message
    if related_question_actions:
        await cl.Message(content="**Related Questions:**", actions=related_question_actions, author="PharmAssistAI").send()

    # Send related papers as a separate message
    if related_papers:
        related_papers_content = "**Related Papers from PubMed:**\n" + "\n".join(f"- {paper}" for paper in related_papers)
        await cl.Message(content=related_papers_content, author="PharmAssistAI").send()

# Action callback for question selection
@cl.action_callback("ask_question")
async def on_question_selected(action: cl.Action):
    """
    Respond to user-selected questions from suggested list, generate and send the answers.
    """
    question = action.value
    await cl.Message(content=question, author="User").send()

    answer, text_elements, related_question_actions, related_papers,query = await generate_answer(question)
    await cl.Message(content=answer, elements=text_elements, author="").send()

    # Send related questions as a separate message
    if related_question_actions:
        await cl.Message(content="**Related Questions:**", actions=related_question_actions, author="").send()

    # Send related papers as a separate message
    if related_papers:
        related_papers_content = "**Related Papers from PubMed:**\n" + "\n".join(f"- {paper}" for paper in related_papers)
        await cl.Message(content=related_papers_content, author="").send()



# Callback for chat start event
@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chatbot environment, load necessary data, and present initial user interactions.
    """
    global qdrant_vectorstore

    # Display a preloader message
    await cl.Message(content="**Loading  PharmAssistAI bot**....").send()
    await asyncio.sleep(2)  # Add a 2-second delay to simulate loading

    # Adding logo for  chatbot
    await cl.Avatar(
        name="",
        url="https://i.imgur.com/ZkIVmxp.jpeg",
    ).send()
  
    # Adding logo for user who is asking questions
    await cl.Avatar(
        name="User",
        url="https://i.imgur.com/XhmbgvT.jpeg",
    ).send()
    
    if qdrant_vectorstore is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")
        QDRANT_CLUSTER_URL =os.environ.get("QDRANT_CLUSTER_URL")
        
        qdrant_client = AsyncQdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY,timeout=60)

        response = await qdrant_client.get_collections()

        # Extracting the collection names from the response
        collection_names = [collection.name for collection in response.collections]

        if "fda_drugs" not in  collection_names:
            print("Collection 'fda_drugs' is not present.")

            # Download the data file
            url = "https://download.open.fda.gov/drug/label/drug-label-0001-of-0012.json.zip"
            response = requests.get(url)
            
            # Extract the JSON file from the zip
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            json_file = zip_file.open(zip_file.namelist()[0])
            
            # Load the JSON data
            data = json.load(json_file)

            df = pd.json_normalize(data['results'])
            selected_drugs = df

            # Define metadata fields to include
            metadata_fields = ['openfda.brand_name', 'openfda.generic_name', 'openfda.manufacturer_name',
                            'openfda.product_type', 'openfda.route', 'openfda.substance_name',
                            'openfda.rxcui', 'openfda.spl_id', 'openfda.package_ndc']

            # Define text fields to index
            text_fields = ['description', 'indications_and_usage', 'contraindications',
                        'warnings', 'adverse_reactions', 'dosage_and_administration']

            # Replace NaN values with empty strings
            selected_drugs[text_fields] = selected_drugs[text_fields].fillna('')

            selected_drugs['content'] = selected_drugs[text_fields].apply(lambda x: ' '.join(x.astype(str)), axis=1)

            loader = DataFrameLoader(selected_drugs, page_content_column='content') 
            drug_docs = loader.load()
            
            for doc, row in zip(drug_docs, selected_drugs.to_dict(orient='records')):
                metadata = {}
                for field in metadata_fields:
                    value = row.get(field)
                    if isinstance(value, list):
                        value = ', '.join(str(v) for v in value if pd.notna(v))
                    elif pd.isna(value):
                        value = 'Not Available'
                    metadata[field] = value
                doc.metadata = metadata  # Update the metadata to only include specified fields
    
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            split_drug_docs = text_splitter.split_documents(drug_docs)

            # Asynchronously create a Qdrant vector store with the document chunks
            qdrant_vectorstore = await cl.make_async(Qdrant.from_documents)(
                split_drug_docs, 
                embedding_model, 
                url=QDRANT_CLUSTER_URL,
                api_key=QDRANT_API_KEY,
                collection_name="fda_drugs"  # Name of the collection in Qdrant
            )
        else:
            print("Collection 'fda_drugs' is present.")
            # Load the existing collection
            qdrant_vectorstore = await  cl.make_async(Qdrant.construct_instance)(
                texts=[""],  # no texts to add
                embedding = embedding_model, 
                url=QDRANT_CLUSTER_URL,
                api_key=QDRANT_API_KEY,
                collection_name="fda_drugs"  # Name of the collection in Qdrant
            )
       

    potential_questions = [
        "What should I be careful of when taking Metformin?",
        "What are the contraindications of Aspirin?", 
        "Are there low-cost alternatives to branded Aspirin available over-the-counter?",
        "What precautions should I take if I'm pregnant or nursing while on Lipitor?",
        "Should Lipitor be taken at a specific time of day, and does it need to be taken with food?",
        "What is the recommended dose of Aspirin?",
        "Can older people take beta blockers?",
        "How do beta blockers work?",
        "Can beta blockers be used for anxiety?",
        "I am taking Aspirin, is it ok to take Glipizide?",
        "Explain in simple terms how Metformin works?"
    ]
    await cl.Message(
        content="**Welcome to PharmAssistAI ! Here are some potential questions you can ask:**",
        actions=[cl.Action(name="ask_question", value=question, label=question) for question in potential_questions]  
    ).send()
    cl.user_session.set("potential_questions_shown", True)

            

                

# Main function to handle user messages
@cl.on_message
async def main(message):
    """
    Process user messages, generate and send responses, and handle further interactions based on the user's queries.
    """
    query = message.content


    try:
        answer, text_elements, related_question_actions, related_papers, original_query = await generate_answer(query)

        # Create a new message with the answer and source documents
        answer_message = cl.Message(content=answer, elements=text_elements, author="PharmAssistAI")
        
        # Send the answer message
        await answer_message.send()
        
        if not answer.lower().startswith("i don't know") and not answer.lower().startswith("i don't have enough information"):
            # Send related questions as a separate message
            if related_question_actions:
                await cl.Message(content="**Related Questions:**", actions=related_question_actions, author="PharmAssistAI").send()

            # Send related papers as a separate message
            if related_papers:
                related_papers_content = "**Related Papers from PubMed:**\n" + "\n".join(f"- {paper}" for paper in related_papers)
                await cl.Message(content=related_papers_content, author="PharmAssistAI").send()

    except Exception as e:
        print(f"Error occurred: {e}")
        answer = "An error occurred while processing your request. Please try again later."
        await cl.Message(content=answer, author="PharmAssistAI").send()