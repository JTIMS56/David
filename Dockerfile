# ── Stage 1: Build the C++ Dirac engine ──────────────────────────────────────
FROM python:3.12-slim AS cpp-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY dirac_engine/ /build/dirac_engine/
WORKDIR /build/dirac_engine
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel 4


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime deps only (libstdc++ for the .so)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy compiled C++ engine
COPY --from=cpp-builder /build/dirac_engine/build/ /app/dirac_engine/build/

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 david && chown -R david:david /app
USER david

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
