FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
# CPU-only torch first: sentence-transformers pulls in torch, and without this it
# grabs the multi-GB GPU/CUDA build by default even though nothing here uses a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
