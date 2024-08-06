FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
#nvcr.io/nvidia/tensorflow:24.07-tf2-py3

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY 2D_CNN_cutout_conv.py /skript.py

COPY helper_funcs.py /helper_funcs.py

ENTRYPOINT [ "python3", "/skript.py" ]