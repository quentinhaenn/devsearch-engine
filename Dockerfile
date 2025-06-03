FROM python:3.12-slim as builder
WORKDIR /app
COPY ./requirements.txt .
RUN pip install -U pip && \
    pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
