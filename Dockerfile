
# write some code to build your image

FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt
COPY api /api
COPY TaxiFareModel /TaxiFareModel
COPY model.joblib /model.joblib

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0
