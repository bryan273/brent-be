FROM python:3.10-alpine3.17

EXPOSE 5000

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD [ "python", "./app.py" ]