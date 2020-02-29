FROM continuumio/miniconda:4.7.12

WORKDIR src

COPY . /src

RUN make install

