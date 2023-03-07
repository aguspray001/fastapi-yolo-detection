FROM tiangolo/uvicorn-gunicorn:python3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/fastapi
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "auto"]
# CMD [ "python", "fastapi/main.py" ]