FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . .

# PYTHONPATH: src/weather_model.py imports config.py from the project root.
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app

CMD ["python", "src/weather_model.py"]
