FROM python:3.11-bookworm@sha256:ba7a7ac30c38e119c4304f98ef0e188f90f4f67a958bb6899da9defb99bfb471 as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=true

RUN pip install --upgrade pip && pip install poetry
RUN poetry config installer.max-workers 10 && \
  poetry config virtualenvs.create false
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"

COPY poetry.lock pyproject.toml ./
RUN poetry install --sync --no-interaction --no-ansi --no-root --only main



ARG TARGETPLATFORM
ENV ARMNN_PATH=/opt/armnn
COPY ann /opt/ann
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    mkdir /opt/armnn && \
    curl -SL "https://github.com/ARM-software/armnn/releases/download/v23.11/ArmNN-linux-aarch64.tar.gz" | tar -zx -C /opt/armnn && \
    cd /opt/ann && \
    sh build.sh; \
  else \
    mkdir /opt/armnn; \
  fi


FROM python:3.11-slim-bookworm@sha256:cfd7ed5c11a88ce533d69a1da2fd932d647f9eb6791c5b4ddce081aedf7f7876
ARG TARGETPLATFORM
RUN apt-get update && apt-get install -y --no-install-recommends tini libmimalloc2.0 && rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends ocl-icd-libopencl1 mesa-opencl-icd && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir --parents /etc/OpenCL/vendors && \
    echo "/usr/lib/libmali.so" > /etc/OpenCL/vendors/mali.icd && \
    mkdir /opt/armnn; \
  fi

WORKDIR /usr/src/app
ENV NODE_ENV=production \
  TRANSFORMERS_CACHE=/cache \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PATH="/opt/venv/bin:$PATH" \
  PYTHONPATH=/usr/src \
  LD_LIBRARY_PATH=/opt/armnn

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/armnn/libarmnn.so.?? /opt/armnn/libarmnnOnnxParser.so.?? /opt/armnn/libarmnnDeserializer.so.?? /opt/armnn/libarmnnTfLiteParser.so.?? /opt/armnn/libprotobuf.so.?.??.?.? /opt/ann/libann.s[o] /opt/ann/build.sh /opt/armnn
COPY ann/ann.py /usr/src/ann/ann.py
COPY start.sh log_conf.json ./
COPY app .

ENTRYPOINT ["tini", "--"]
CMD ["./start.sh"]
