ARG IMAGE_DISTRO=12.2.2-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:${IMAGE_DISTRO} AS builder

WORKDIR /build

COPY . /build/

RUN make

CMD ["./gpu_burn", "60"]
