import os

IP_WHITELIST = os.environ.get("IP_WHITELIST") or ["127.0.0.1"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH") or "./vector_stores/"

# AWS
AWS_ACCESS_KEY_ID = os.getenv("MYACCESSKEY")
AWS_SECRET_ACCESS_KEY = os.getenv("MYSECRETKEY")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"

PORT = os.getenv("PORT") or 8000