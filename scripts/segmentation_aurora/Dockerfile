FROM nvcr.io/nvidia/cuda:12.5.0-devel-rockylinux9

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY segmentation_aurora.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]