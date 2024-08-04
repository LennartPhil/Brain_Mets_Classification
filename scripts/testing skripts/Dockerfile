FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

#RUN apt-get install -y nvidia-container-toolkit

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY tuner_testing.py /skript.py
#COPY 3D_CNN_test.py /skript.py

#COPY helper_funcs.py /helper_funcs.py

#ENTRYPOINT [ "python3", "/skript.py" ]
CMD tail -f  /dev/null