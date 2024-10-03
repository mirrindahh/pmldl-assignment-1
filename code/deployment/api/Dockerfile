FROM python:3.10-slim

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./cats-and-dogs.pt ./cats-and-dogs.pt
COPY ./app.py ./app.py

ENTRYPOINT ["python", "app.py"]
