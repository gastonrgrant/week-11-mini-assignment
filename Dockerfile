FROM python:3.12-slim

WORKDIR /repo

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/repo \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agents.server:app", "--host", "0.0.0.0", "--port", "8000"]
