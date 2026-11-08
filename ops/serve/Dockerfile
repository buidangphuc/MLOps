FROM python:3.12-slim
ARG PACK=text_classification
WORKDIR /app

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY ops/serve/requirements.txt shell-requirements.txt
COPY packs/${PACK}/serving-requirements.txt pack-requirements.txt
RUN pip install --no-cache-dir -r shell-requirements.txt -r pack-requirements.txt

COPY ops/serve/app ./app
COPY packs/__init__.py ./packs/__init__.py
COPY packs/${PACK} ./packs/${PACK}

ENV DS_PACK=${PACK} \
    CUDA_VISIBLE_DEVICES="" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app

EXPOSE 8080
CMD ["python", "-m", "app.main"]
