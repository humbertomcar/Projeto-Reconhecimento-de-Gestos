FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install -y \
    git vim \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libv4l-0 v4l-utils \
  && rm -rf /var/lib/apt/lists/*
  
COPY requirements.txt ./

RUN python -m pip install --upgrade pip  && python -m pip install -r requirements.txt