FROM python:3.7

ADD . /clinical_trials
RUN mkdir /clinical_trials/data /clinical_trials/logs

WORKDIR /clinical_trials
RUN pip3 install -r requirements.txt
CMD ["python3", "wrapper.py"]
