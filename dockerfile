FROM python:3.8

WORKDIR /
RUN pip install -r requirements.txt


CMD ["python3", "app.py"]