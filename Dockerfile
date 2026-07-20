FROM python:3.11-slim

WORKDIR /app

# Copiamos primero el requirements.txt para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código y los modelos
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
