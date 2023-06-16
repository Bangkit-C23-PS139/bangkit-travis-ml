FROM python:3.9.17-buster

WORKDIR /ml

COPY . /ml/

RUN pip install -r requirements.txt

EXPOSE 8080

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]