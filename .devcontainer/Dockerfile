# 使用 Python 3.12 的官方基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製當前目錄內容到容器的工作目錄
COPY . .

# 安裝必要的 Python 套件
# RUN pip install --no-cache-dir -r requirements.txt

# 設定容器啟動時執行的命令
CMD ["python", "app.py"]