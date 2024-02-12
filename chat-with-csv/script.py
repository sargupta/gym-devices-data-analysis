from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers



DB_FAISS_PATH = "vectorstore/db_faiss"
# Load the data
loader = CSVLoader(file_path="data/2019.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
# print(data)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# print(len(text_chunks))

#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Download sentence transformers embedding from huggingface 
# embedding_model = HuggingFaceEmbedding(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)

query = "what's the GDP of Finland?"

docs = docsearch.similarity_search(query, k=4)

#print("Result", docs)

llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    # query = "what's the GDP of Finland?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
        
    if query == '':
        continue
    result = qa({"question": query, "chat_history": chat_history})
    print("Response: ", result['answer'])

