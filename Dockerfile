FROM python:3.10-slim

# Ustawiamy katalog roboczy, w którym będzie działać aplikacja
WORKDIR /app

# Najpierw kopiujemy tylko plik zależności
# Pozwala to wykorzystać cache Dockera przy przebudowie obrazu
COPY requirements.txt .

# Instalujemy zależności z requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Teraz kopiujemy resztę projektu
COPY . .

# Eksponujemy port, na którym działa API
EXPOSE 8000

# Komenda uruchamiająca serwer FastAPI (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]