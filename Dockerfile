FROM python:3.7

ENV PYTHONUNBUFFERED 1

EXPOSE 5010


COPY . /inference-api
WORKDIR /inference-api/

#RUN apt-get update
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#COPY ./models.tar.gz /inference-api/
#RUN tar -zxf /inference-api/models.tar.gz --directory /inference-api
#RUN rm /inference-api/models.tar.gz

RUN pip3 install -r /inference-api/requirements.txt

CMD [ "python", "/inference-api/server.py" ]