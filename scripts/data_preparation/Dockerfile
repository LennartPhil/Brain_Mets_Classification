#FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3-igpu
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

COPY requirements.txt /requirements.txt

RUN apt-get update

RUN apt-get install -y python3-pip

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY data_preparation.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]