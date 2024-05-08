FROM python:3.8-slim-buster

WORKDIR /src

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/weather_model.py"]