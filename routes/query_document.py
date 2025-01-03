from fastapi import APIRouter, Query, File, UploadFile
from services.rag import load_document, split_document, embedder_by_huggingface, create_or_load_vector_store
from utils.document_comparison import add_unique_documents
from services.llm_chain import inference_chain_rag
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
from services.bedrock import bedrock_llm, bedrock_embeddings, get_bedrock_client
query_document_router = APIRouter()

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"

@query_document_router.post("/query_document")
async def query_document(
    query: str = Query(..., description="Query for the document"),
    vector_store_name: str = Query(..., description="Name of the vector store"),
    vector_store_type: VectorStoreType = Query(..., description="Type of the vector store"),
    file: UploadFile = File(..., description="File to be queried"),
):
    documents = load_document(file)
    documents = split_document(documents)
    bedrock_client = get_bedrock_client()
    embeddings = bedrock_embeddings(
        model_id="amazon.titan-embed-text-v2:0",
        bedrock_client=bedrock_client
    )
    llm = bedrock_llm(bedrock_client=bedrock_client)

    vector_store = create_or_load_vector_store(
        embeddings=embeddings,
        vector_store_type=vector_store_type,
        vector_store_name=vector_store_name,
        documents=documents
    )
    add_unique_documents(documents, vector_store, vector_store_type, vector_store_name)
    template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=template)
    chain = inference_chain_rag(
        vectorstorage=vector_store, 
        llm=llm,
        prompt_template=prompt_template,
    )
    response = chain.invoke({"input": query})
    return {"chain_response": response["answer"]}