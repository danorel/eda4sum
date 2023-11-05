FROM python:3.8

EXPOSE 8080

COPY ./app /app
COPY ./client /client
COPY ./rl /rl
COPY ./requirements.txt /
COPY ./galaxy_classes_mean_vectors.json /

RUN pip install -r /requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
