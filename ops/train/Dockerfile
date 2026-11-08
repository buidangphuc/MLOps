FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /opt/program

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY packs ./packs
COPY config ./config
COPY main.py ./
RUN pip install --no-cache-dir -e ".[dev,gpu]"

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["python", "main.py"]
