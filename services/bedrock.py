import boto3
from langchain_aws import BedrockLLM, BedrockEmbeddings
from constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def get_bedrock_client(access_key_id: str = AWS_ACCESS_KEY_ID, secret_access_key: str = AWS_SECRET_ACCESS_KEY):
    bedrock_client=boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
    return bedrock_client

def bedrock_embeddings(model_id: str = "amazon.titan-embed-text-v1", bedrock_client: boto3.client = None):
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=model_id,
        model_kwargs={"dimensions": 256}
    )
    return embeddings

def bedrock_llm(model_id: str = "us.meta.llama3-1-8b-instruct-v1:0", bedrock_client: boto3.client = None):
    llm = BedrockLLM(
        client=bedrock_client,
        model_id=model_id,
        max_tokens=512
    )
    return llm


