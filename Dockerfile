# minimal python image
FROM python:3.10-slim

# system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# set the working directory
WORKDIR /app

# load in requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add in the what the app needs from src
COPY app.py ./
COPY src ./src
COPY deploy ./deploy
COPY models ./models
COPY configs ./configs
COPY feedback ./feedback
COPY demo_images ./demo_images
COPY feedback ./feedback

# environments
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# start streamlit in cloud run port
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]

