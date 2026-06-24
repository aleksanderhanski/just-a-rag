FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Byte-compile for faster startup; copy mode avoids cross-filesystem link warnings
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

WORKDIR /app/src

EXPOSE 7860

CMD ["uv", "run", "app.py"]