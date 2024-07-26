FROM python:3.10

WORKDIR /stocknear-backend

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .


CMD ["uvicorn", "main:app", "--reload"]