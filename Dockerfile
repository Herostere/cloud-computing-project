FROM python:3.10

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . code
WORKDIR /code

EXPOSE 6660

ENTRYPOINT ["python3", "Project/backend/manage.py"]
CMD ["runserver", "0.0.0.0:6660"]
