FROM python:3.10-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install -r ./requirements.txt

COPY . .

ENTRYPOINT ["python", "runner.py"]
