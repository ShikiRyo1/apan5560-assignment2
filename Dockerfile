# docker build -t assignment2 .
# docker run --rm --name assignment2 -p 127.0.0.1:8000:8000 assignment2
# Open http://127.0.0.1:8000/docs; POST /classify accepts a PNG or JPEG upload.
# Dependencies and both pretrained models are bundled during the build.
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
COPY app ./app
COPY models ./models
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
