FROM python:slim-buster

COPY flask_app.py flask_app.py
COPY hockey_classifier.py hockey_classifier.py
COPY hockey.txt hockey.txt
COPY sports.txt sports.txt
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "flask_app.py"]