FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
#nvcr.io/nvidia/tensorflow:24.07-tf2-py3

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

#COPY 2D_CNN_cutout_conv.py /conv_skript.py
#COPY 2D_CNN_cutout_resnet34.py /resnet34_skript.py
COPY 2D_CNN_cutout_resnet152.py /resnet152_skript.py
COPY 2D_CNN_cutout_resnext50.py /resnext50_skript.py
COPY 2D_CNN_cutout_resnext101.py /resnext101_skript.py
COPY 2D_CNN_cutout_pretrained_inceptionv3.py /inceptionv3_skript.py
COPY 2D_CNN_cutout_pretrained_resnet50v2.py /resnet50v2_skript.py

COPY helper_funcs.py /helper_funcs.py

ENTRYPOINT [ "sh", "-c", "python3 /resnet152_skript.py && python3 /resnext50_skript.py && python3 /resnext101_skript.py && python3 /inceptionv3_skript.py && python3 /resnet50v2_skript.py" ]

#ENTRYPOINT [ "python3", "/skript.py" ]-