FROM python:3.7

ADD . /clinical_trails
RUN mkdir data logs

WORKDIR /clinical_trails
#RUN pip install -r requirements.txt
CMD ["python3", "wrapper.py"]
