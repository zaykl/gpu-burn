ARG IMAGE_DISTRO=12.2.2-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:${IMAGE_DISTRO} AS builder

RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN  apt-get clean

RUN mkdir /run/sshd
RUN apt-get update && apt-get install -y \
    systemctl \
    vim \
    openssh-server

WORKDIR /build

COPY . /build/

RUN make

CMD ["./gpu_burn", "60"]
