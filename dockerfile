FROM python:3.7

ADD . /clinical_trails
RUN mkdir /clinical_trails/data /clinical_trails/logs

WORKDIR /clinical_trails
RUN pip3 install -r requirements.txt
CMD ["python3", "wrapper.py"]
