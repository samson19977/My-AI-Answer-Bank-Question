from datasets import load_dataset
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# Load the dataset
ds = load_dataset("maxpro291/bankfaqs_dataset")
train_ds = ds['train']

# Convert the dataset into a pandas DataFrame
data = train_ds[:]
questions = []
answers = []

# Parse the 'text' field to separate questions and answers
for entry in data['text']:
    if entry.startswith("Q:"):
        questions.append(entry)
    elif entry.startswith("A:"):
        answers.append(entry)

# Create a DataFrame with only 'question' and 'answer' columns
Bank_Data = pd.DataFrame({'question': questions, 'answer': answers})

# Prepare context data for vector storage
context_data = []
for i in range(len(Bank_Data)):
    context = ""
    for j in range(min(2, len(Bank_Data.columns))):
        context += Bank_Data.columns[j] + ": "
        context += str(Bank_Data.iloc[i, j]) + " "
    context_data.append(context)

# Get the secret key from the environment
groq_key = os.environ.get('Samson')

# Initialize the LLM for RAG
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_key)

# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Create vector store
vectorstore = Chroma(
    collection_name="bank_store",
    embedding_function=embed_model,
    persist_directory="./",
)
# Add data to the vector store
vectorstore.add_texts(context_data)

# Create a retriever
retriever = vectorstore.as_retriever()

# Create a Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Define the prompt template
template = ("""You are a Customer Care.
    Use the provided context to answer the question.
    If you don't know the answer, say so. Explain your answer in detail.
    Do not discuss the context in your response; just provide the answer directly.
    Context: {context}
    Question: {question}
    Answer:""")

rag_prompt = PromptTemplate.from_template(template)

# Build the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Define the Gradio ChatInterface
def rag_memory_stream(message, history):
    partial_text = ""
    for new_text in rag_chain.stream(message):
        partial_text += new_text
        yield partial_text

examples = [
    "I want to open an account", 
    "What is a savings account?"
]

title = "Your Personal Banking Assistant 💬"
description = (
    "Welcome! 👋 I’m here to make banking super easy and stress-free. "
    "Have a question about accounts, services, or banking processes? Ask away! "
    "No waiting in lines or long processes—just ask me and get immediate help! 🌟"
)

# Use ChatInterface for a chat-style UI
demo = gr.ChatInterface(
    fn=rag_memory_stream,
    title=title,
    description=description,
    examples=examples,
    theme="glass",
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
