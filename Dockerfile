FROM continuumio/miniconda3

WORKDIR /home
ENV PYTHONPATH=/home

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

COPY . .
RUN pip install -r requirements.txt

CMD ["python", "app/train.py"]