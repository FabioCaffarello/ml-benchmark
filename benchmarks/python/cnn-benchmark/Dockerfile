FROM python:3.12-slim


COPY dist/*.tar.gz .
RUN pip install *.tar.gz
RUN rm -rf *.tar.gz


WORKDIR /app
COPY cnn_benchmark cnn_benchmark


ENTRYPOINT ["python", "/app/cnn_benchmark/main.py"]
