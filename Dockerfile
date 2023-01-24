# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx libxcb-xinerama0

WORKDIR /golfdb

RUN pip install -r requirements.txt