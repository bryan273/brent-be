FROM python:3.10-slim-buster

EXPOSE 5000

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD [ "python", "./app.py" ]