# Multi-arch Docker image for the simulation API using FEniCS via conda-forge
# Works on Apple silicon by leveraging conda packages

FROM mambaorg/micromamba:1.5.7

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

# Install FEniCS and runtime deps from conda-forge
# Also install FastAPI stack and plotting libs
RUN micromamba install -y -n base -c conda-forge \
    fenics=2019.1.* \
    python=3.11 \
    fastapi=0.111 \
    uvicorn=0.30 \
    numpy \
    matplotlib \
    plotly \
    && micromamba clean -a -y

# Copy backend code
COPY backend /app/backend

EXPOSE 8000

# Run API under conda environment
CMD ["micromamba", "run", "-n", "base", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]


