FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
#FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3
#nvcr.io/nvidia/tensorflow:24.07-tf2-py3

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip

COPY regular_requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

#COPY 2D_CNN_conv.py /conv_skript.py
#COPY 2D_CNN_resnet34.py /resnet34_skript.py
COPY 2D_CNN_resnet152.py /resnet152_skript.py
COPY 2D_CNN_resnext50.py /resnext50_skript.py
COPY 2D_CNN_resnext101.py /resnext101_skript.py
#COPY 2D_CNN_transfer_bit.py /bit_skript.py
#COPY 2D_CNN_transfer_efficientv2.py /efficientv2_skript.py
#COPY 2D_CNN_transfer_inceptionv3.py /inceptionv3_skript.py
#COPY 2D_CNN_transfer_resnet50v2.py /resnet50v2_skript.py
#COPY 2D_transfer_vit.py /vit_skript.py

COPY helper_funcs.py /helper_funcs.py
COPY constants.py /constants.py

# run all scripts
#ENTRYPOINT [ "sh", "-c", "python3 /conv_skript.py && python3 /resnet34_skript.py && python3 /resnet152_skript.py && python3 /resnext50_skript.py && python3 /resnext101_skript.py && python3 /bit_skript.py && python3 /inceptionv3_skript.py && python3 /resnet50v2_skript.py"]

# run pretrained scripts
#ENTRYPOINT [ "sh", "-c", "python3 /bit_skript.py && python3 /inceptionv3_skript.py && python3 /resnet50v2_skript.py"]

ENTRYPOINT [ "sh", "-c", "python3 /resnet152_skript.py && python3 /resnext101_skript.py" ]
#ENTRYPOINT [ "sh", "-c", "python3 /resnet152_skript.py && python3 /resnext50_skript.py && python3 /resnext101_skript.py" ]
#ENTRYPOINT [ "python3", "/skript.py" ]-