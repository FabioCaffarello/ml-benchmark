FROM python:3.12-slim


COPY dist/*.tar.gz .
RUN pip install *.tar.gz
RUN rm -rf *.tar.gz


WORKDIR /app
COPY downloader downloader
COPY hack hack

RUN chmod +x /app/hack/entrypoint.sh

RUN mkdir -p /app/data


ENTRYPOINT ["/app/hack/entrypoint.sh"]
