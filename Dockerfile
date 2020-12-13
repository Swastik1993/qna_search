# build stage
FROM ubuntu:latest AS compile-image
# update and install necessary packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc python3 python3-venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# install necessary python libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# final stage
FROM ubuntu:latest AS build-image
# update and install necessary packages
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk
RUN apt-get install -y --no-install-recommends curl build-essential gcc python3 python3-venv
# copy build libraries and code from compile image
WORKDIR /code
COPY --from=compile-image /opt/venv /opt/venv
COPY ./src .
# setting virtual environment path
ENV PATH="/opt/venv/bin:$PATH"
# load data into docker image
RUN mkdir -p /data/indexes/
COPY data/lucene-index-cord19.tar.gz /data/indexes/
RUN tar -xvzf /data/indexes/lucene-index-cord19.tar.gz -C /data/indexes/
RUN rm -R /data/indexes/lucene-index-cord19.tar.gz
# download and store all required models directly inside docker image
RUN python init_models.py
# run python app
CMD [ "uvicorn", "--host", "0.0.0.0", "app:app" ]
