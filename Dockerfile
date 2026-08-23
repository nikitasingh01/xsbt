# Clean-room run of the backtester. Only `xsbt fetch` needs the network; everything
# else replays from a mounted cache, so this can run in a locked-down environment.
FROM python:3.13-slim

# PYTHONUNBUFFERED so `docker logs` shows progress while a fetch is running rather than
# after it finishes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependency metadata first, so a source change does not invalidate the pip layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

# Nothing here needs root, and a container that writes into a mounted volume as root
# leaves root-owned files behind on the host.
RUN useradd --create-home --uid 10001 xsbt \
    && mkdir -p /app/data /app/runs /app/reports \
    && chown -R xsbt:xsbt /app
USER xsbt

VOLUME ["/app/data", "/app/runs"]

ENTRYPOINT ["xsbt"]
CMD ["--help"]
