FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencia del sistema necesaria para soundfile/libsndfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias
COPY backend/pyproject.toml backend/uv.lock ./backend/

WORKDIR /app/backend

# Instalar dependencias sin intentar instalar el proyecto como paquete
RUN uv sync --frozen --no-dev --no-install-project

WORKDIR /app

# Copiar backend y detector
COPY backend ./backend
COPY detector ./detector

# Permitir importar backend desde src
ENV PYTHONPATH=/app/backend/src

EXPOSE 8000

CMD ["/app/backend/.venv/bin/python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]