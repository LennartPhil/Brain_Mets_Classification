FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY segmentation_aurora.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]