FROM nvcr.io/nvidia/ai-workbench/python-basic:1.0.2

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY preprocessing_brainlesion_n4.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]