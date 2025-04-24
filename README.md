# Chatbot-baseado-em-conte-do-de-PDFs

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
  from langchain.chains import RetrievalQA 
 loader = PyPDFLoader("Curriculo -2025.pdf") 
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter( 
 chunk_size=1000, chunk_overlap=200 ) 
splits = text_splitter.split_documents(pages)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings) 
llm = ChatOpenAI(temperature=0) qa_chain = RetrievalQA.from_chain_type( llm, retriever=vectorstore.as_retriever() ) 
def chat_with_pdf(question): result = qa_chain({"query": question}) return result["result"]
 print(chat_with_pdf("Qual é o tema principal deste documento?"))