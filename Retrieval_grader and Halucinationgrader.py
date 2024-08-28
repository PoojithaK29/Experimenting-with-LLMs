import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Assuming you named the model "llama3" during installation (modify if different)
local_llm = Ollama(model="llama3")
# Adjust these paths to the locations of your PDF files
pdf_paths = [
    "C:/Users/pkathi/Downloads/GSP-QU-H01-0001 - B02 - QRQC Working Method.docx.pdf",
    "C:/Users/pkathi/Downloads/PEM-QUAL-SSQG-001 V5.8.docx.pdf",
]

# Load PDF documents
documents = [PDFMinerLoader(path).load() for path in pdf_paths]
docs_list = [item for sublist in documents for item in sublist]  # Flatten the list

# Text chunking (adjust chunk_size and chunk_overlap if needed)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Create Chroma vectorstore with your preferred embedding model
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# LLM model
local_llm = "llama3"
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Retrieval Grader Prompt
retrieval_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

# Hallucination Grader Prompt
hallucination_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
from langchain.prompts import PromptTemplate

# Then use PromptTemplate as usual:

qa_prompt = """... Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
|eot_id|><|start_header_id|>user<|end_header_id|>
... Question: {question}
... Context: {context}
... Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


qa_chain = qa_prompt | llm | StrOutputParser()

# Conversation history (to store past interactions)
conversation_history = []

def main():


    # Text box for user question with history display (optional)
    user_question = st.text_input("Ask me anything!", key="user_question")

    if conversation_history:
        st.header("Conversation History")
        for i, entry in enumerate(conversation_history):
            st.write(f"{i+1}. {entry['question']}")
            st.write(f"**Answer:** {entry['answer']}", background_color="white")
            st.write("---")

    # Run the system if user enters a question
    if user_question:
        # Retrieve documents based on the question
        retrieved_docs = retriever.invoke(user_question)

        # Filter documents based on retrieval grader (optional)
        filtered_docs = retrieved_docs
        # if you want to use retrieval grader, uncomment the following lines
        # for doc in retrieved_docs:
        #     score_output = retrieval_grader.invoke({"question": user_question, "document": doc.page_content})
        #     if score_output['score'] == 'yes':
        #         filtered_docs.append(doc)

        # Format context
        context = "\n\n".join(doc.page_content for doc in filtered_docs)

        # Process the question through QA pipeline
        generation = qa_chain.invoke({"context": context, "question": user_question})

        # Perform hallucination grading (optional)
        hallucination_score = None
        # if you want to use hallucination grader, uncomment the following lines
        # hallucination_output = hallucination_grader.invoke({"generation": generation, "documents": context})
        # hallucination_score = hallucination_output['score']

        # Update conversation history
        conversation_history.append({"question": user_question, "answer": generation})

        # Display answer with visual formatting
        st.success(f"Answer: {generation}", background_color="white")

        # Display hallucination score (optional)
        if hallucination_score:
            st.write(f"Hallucination Score: {hallucination_score}")

# Streamlit app execution
if __name__ == "__main__":
    main()
