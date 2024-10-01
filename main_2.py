from flask import Flask, render_template, request, jsonify
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load the text from the file with explicit encoding
try:
    with open('pdf_text2', 'r', encoding='utf-8') as file:
        prompt = file.read()
except FileNotFoundError:
    print("The prompt file 'pdf_text' was not found.")
    exit(1)
except UnicodeDecodeError:
    print("Could not decode 'pdf_text'. Please ensure it is encoded in UTF-8.")
    exit(1)

# Limit the prompt length if it's too long
max_prompt_length = 3000
chunks = [prompt[i:i + max_prompt_length] for i in range(0, len(prompt), max_prompt_length)]

# Semantic search setup with FAISS 
# What is FAISS? FAISS is a library for efficient similarity search and clustering of dense vectors.
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embeddings)

# Define a function to get relevant text based on a question
def get_relevant_text(question):
    results = vector_store.similarity_search(question, k=3)
    return " ".join([doc.page_content for doc in results])  # Extract text from Document objects

# Define the book assistant template
book_assistant_template = """
You are an assistant specialized in providing insights and explanations based on the book 'The Mountain Is You' by Brianna Wiest. 
Your role is to discuss the book's themes, concepts, and teachings, including personal growth, emotional transformation, and overcoming self-sabotage. 
If a question is not related to the book, respond with, "I can't assist you with that, sorry!" 
Question: {question} 
Answer: 
"""

# Create the prompt template
book_assistant_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=book_assistant_template
)

# Set up the language model
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, max_tokens=256)

# Use the pipe operator for creating the sequence
llm_chain = book_assistant_prompt_template | llm

# Define a function to query the language model
# This function will return the response based on the question
def query_llm(question):
    try:
        relevant_text = get_relevant_text(question)
        prompt_template = relevant_text + book_assistant_template
        book_assistant_prompt_template.template = prompt_template
        response = llm_chain.invoke({'question': question})
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "There was an error processing your request."

# Initialize Flask app
# This will create a simple web interface for interacting with the chatbot
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data.get("question", "")
    response = query_llm(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

