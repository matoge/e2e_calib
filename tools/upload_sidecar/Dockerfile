FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.3 werkzeug==3.0.3
COPY server.py /app/server.py
COPY static /app/static
ENV UPLOAD_DIR=/data
EXPOSE 5001
CMD ["python", "/app/server.py"]
