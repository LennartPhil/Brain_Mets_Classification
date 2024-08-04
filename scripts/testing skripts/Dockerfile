FROM tensorflow/tensorflow:latest-gpu
#nvcr.io/nvidia/tensorflow:24.07-tf2-py3

RUN apt-get update

RUN apt-get install -y nvidia-container-toolkit

RUN apt-get install -y python3-pip

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY tuner_testing.py /skript.py
#COPY 3D_CNN_test.py /skript.py

#COPY helper_funcs.py /helper_funcs.py

ENTRYPOINT [ "python3", "/skript.py" ]