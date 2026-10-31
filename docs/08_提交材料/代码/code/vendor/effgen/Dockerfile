FROM python:3.11-slim

LABEL maintainer="Gaurav Srivastava <gks@vt.edu>"
LABEL description="effGen — Agentic AI framework for Small Language Models"
LABEL version="0.2.0"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY . .
RUN pip install --no-cache-dir -e .

# Verify install
RUN python -c "import effgen; print(f'effGen {effgen.__version__} installed')"
RUN python -c "from effgen.guardrails import GuardrailChain; print('Guardrails OK')"
RUN python -c "from effgen.rag import DocumentIngester; print('RAG OK')"
RUN python -c "from effgen.eval import AgentEvaluator; print('Eval OK')"
RUN python -c "from effgen.domains import TechDomain; print('Domains OK')"

# Default: start the API server
EXPOSE 8000
ENV EFFGEN_RATE_LIMIT=60

ENTRYPOINT ["effgen"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
