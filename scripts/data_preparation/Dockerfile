FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3-igpu

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY data_preparation.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]