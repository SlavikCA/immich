ARG DEVICE=cpu
FROM python:3.11-bookworm@sha256:ba7a7ac30c38e119c4304f98ef0e188f90f4f67a958bb6899da9defb99bfb471 as builder-cpu
FROM python:3.10-bookworm@sha256:77335ff30c2b89e66ae3b70589ca20a3f3cf1c815710438d14d61661b37d1a4a as builder-openvino
FROM builder-cpu as builder-cuda

FROM builder-${DEVICE} as builder
ARG DEVICE=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=true

RUN pip install --upgrade pip && pip install poetry
RUN poetry config installer.max-workers 10 && \
  poetry config virtualenvs.create false
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"

COPY poetry.lock pyproject.toml ./
RUN poetry install --sync --no-interaction --no-ansi --no-root --with ${DEVICE} --without dev

FROM python:3.11-slim-bookworm@sha256:cfd7ed5c11a88ce533d69a1da2fd932d647f9eb6791c5b4ddce081aedf7f7876 as prod-cpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini libmimalloc2.0 && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.10-slim-bookworm@sha256:a2c9b8dd3da225debeb156176c111752499d152b7505146c1373364766d762a4 as prod-openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini libmimalloc2.0 && \
    rm -rf /var/lib/apt/lists/*

FROM nvcr.io/nvidia/tensorrt:23.11-py3@sha256:17e7db50fb0605dd3818b3460655d5a840462a2b12a725d36053f5765a6b98ff as prod-cuda
RUN apt-get update && \
    apt-get install -yqq software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -yqq --no-install-recommends tini libmimalloc2.0 python3.11 && \
    apt-get remove -yqq software-properties-common && \
    apt-get autoremove -yqq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM prod-${DEVICE} as prod

WORKDIR /usr/src/app
ENV NODE_ENV=production \
  TRANSFORMERS_CACHE=/cache \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PATH="/opt/venv/bin:$PATH" \
  PYTHONPATH=/usr/src

COPY --from=builder /opt/venv /opt/venv
COPY start.sh log_conf.json ./
COPY app .
ENTRYPOINT ["tini", "--"]
CMD ["./start.sh"]
