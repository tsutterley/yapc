FROM python:3.9-slim-buster

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Los_Angeles"

RUN useradd --create-home --shell /bin/bash yapc

RUN apt-get update -y && \
    apt-get install -y \
        gcc \
        git \
        libhdf5-dev && \
    apt-get clean

WORKDIR /home/yapc

RUN pip3 install --no-cache-dir  \
        cython \
        numpy \
        h5py \
        scikit-learn \
        future

COPY . .

RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir --no-deps .

USER yapc

CMD ["bash"]
