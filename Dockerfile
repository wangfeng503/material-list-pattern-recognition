FROM python:3.11-slim

WORKDIR /app

# 使用阿里云 Debian 镜像源（提升国内构建速度，ARM 平台同样适用）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

COPY config.yaml .
COPY dataset/lumian/ ./dataset/lumian/
COPY 5matching_system.py .
COPY app.py .

EXPOSE 5000

# 使用 Python 进行健康检查（无需安装 curl）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/predictor/health')" || exit 1

CMD ["gunicorn", "--workers=4", "--timeout=60", "--bind=0.0.0.0:5000", "--access-logfile=-", "--error-logfile=-", "app:app"]
