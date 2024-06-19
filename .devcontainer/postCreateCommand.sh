#!/bin/sh

# Setting node
npm install || { echo "npm install failed"; exit 1; }

echo "npm install succeeded"


# Setting husky
npm run prepare || { echo "npm run prepare failed"; exit 1; }

echo "npm run prepare succeeded"


# Setting golang
# Documentation
go install github.com/princjef/gomarkdoc/cmd/gomarkdoc@latest || { echo "go install failed"; exit 1; }

echo "go install succeeded"


# Setting python
pip install poetry || { echo "pip install poetry failed"; exit 1; }
poetry install || { echo "poetry install failed"; exit 1; }

echo "poetry install succeeded"


echo "All setup succeeded"