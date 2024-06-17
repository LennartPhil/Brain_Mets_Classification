FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

RUN apt-get update

RUN apt-get install -y python3-pip

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY 3D_CNN_test.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]