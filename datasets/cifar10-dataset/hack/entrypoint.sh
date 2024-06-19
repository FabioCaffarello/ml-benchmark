#!/bin/sh

rm -rf /app/data/*

python downloader/main.py

touch /app/data/processing_complete

tail -f /dev/null