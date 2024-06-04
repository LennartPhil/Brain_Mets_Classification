FROM python:3.11.9-slim-bullseye

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY preprocessing_brainlesion_n4.py /skript.py

ENTRYPOINT [ "python3", "/skript.py" ]