# 베이스 이미지 선택 (Python 3.12 slim 버전)
FROM python:3.12.6-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcups2 \
    libgtk-3-0 \
    libxshmfence1 \
    libmagic1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    pandoc \
    qpdf

# Google Chrome의 GPG 키 추가 및 리포지토리 설정
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list

# Google Chrome 설치
RUN apt-get update && apt-get install -y google-chrome-stable

# 필요한 패키지 복사 및 설치
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

# 프로젝트의 전체 소스 코드 복사
COPY . .

# PYTHONPATH 설정 (내부 모듄 인식을 위해)
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 애플리케이션 실행 명령어 설정
CMD ["python", "crawler/wiki.py"]
