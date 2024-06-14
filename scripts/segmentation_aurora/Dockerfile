FROM nvcr.io/nvidia/cuda:12.5.0-devel-ubuntu22.04

RUN apt-get update

RUN apt-get install -y python3-pip

RUN pip install --upgrade pip

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY segmentation_aurora.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]