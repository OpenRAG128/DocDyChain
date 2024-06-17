import os
import PyPDF2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

groq_api_key = os.getenv("GROQ_API_KEY")
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.2
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf = PyPDF2.PdfReader(file.file)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return JSONResponse({"message": "File processed successfully", "chain_id": id(chain)})

@app.post("/ask")
async def ask_question(chain_id: int, question: str):
    chain = ... # retrieve the chain object using the chain_id
    res = await chain.ainvoke(question)
    answer = res["answer"]
    source_documents = res["source_documents"]
    sources = [doc.page_content for doc in source_documents]
    return {"answer": answer, "sources": sources}
