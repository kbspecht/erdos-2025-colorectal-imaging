# 🩺 Colorectal Polyp Detection App

This repository contains a Streamlit application for **colorectal polyp detection**, packaged for local use with Docker and deployment on **Google Cloud Run**.

---

## Deployment

### 0. Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python 3.10+** (for local development; optional if you only use Docker)
- **[Docker](https://www.docker.com/)** — installed and running
- **Google Cloud SDK (`gcloud`)** — installed and authenticated
- A **Google Cloud project**, e.g.:
  ```bash
  gcloud init
  gcloud auth login
  gcloud config set project polyp-detection-demo
  ```

---

### 1. Run Locally (Without Docker)

You can run the Streamlit app directly in a local Python environment.

```bash
# Create and activate a virtual environment
conda create -n polyp-app python=3.10 -y
conda activate polyp-app

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

Then open your browser to:  
👉 http://localhost:8080

---

### 2. Run Locally With Docker

Build and run the app in an isolated Docker container.

#### 2.1 Build the Docker image
From the repository root (where your `Dockerfile` is located):

```bash
docker build -t polyp-app:v1 .
```

> You can replace `v1` with any version tag you prefer.

#### 2.2 Run the container

```bash
docker run -p 8080:8080 polyp-app:v1
```

Now visit:  http://localhost:8080

If your app needs environment variables (e.g. model paths or configuration):

```bash
docker run -p 8080:8080   -e MODEL_PATH=/app/models/best.pth   polyp-app:v1
```

---

### 3. Deploy to Google Cloud Run

Deploy your containerized app to **Google Cloud Run** for serverless hosting.

#### 3.1 Build and Push the Image to Google Container Registry (GCR)

```bash
VERSION=v1

gcloud builds submit   --tag gcr.io/polyp-detection-demo/polyp-app:$VERSION .
```

This command:
- Builds the Docker image using your local `Dockerfile`
- Pushes it to `gcr.io/polyp-detection-demo/polyp-app:$VERSION`

#### 3.2 Deploy the Image to Cloud Run

```bash
gcloud run deploy polyp-app   --image gcr.io/polyp-detection-demo/polyp-app:$VERSION   --platform managed   --region europe-west1   --allow-unauthenticated   --port 8080
```

- `--allow-unauthenticated` makes the app public  
- `--port 8080` matches the Streamlit app’s internal port  

After deployment, you’ll see output like:

```
Service [polyp-app] revision [polyp-app-00001-...] has been deployed and is serving 100 percent of traffic at URL:
https://polyp-app-xxxxx-uc.a.run.app
```

Open that URL in your browser to access the deployed app.

---

### Notes

- **Dockerfile** should expose port 8080 and include:
  ```dockerfile
  CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
  ```
- **.dockerignore** should exclude large files and unnecessary directories (e.g. data, checkpoints, cache).
- To delete the deployed service when not in use:
  ```bash
  gcloud run services delete polyp-app --region=europe-west1
  ```

---

### ✅ Summary

| Step | Command | Description |
|------|----------|-------------|
| Build Docker | `docker build -t polyp-app:v1 .` | Creates the local Docker image |
| Run Docker | `docker run -p 8080:8080 polyp-app:v1` | Runs locally on port 8080 |
| Push to GCR | `gcloud builds submit --tag gcr.io/polyp-detection-demo/polyp-app:v1 .` | Builds and uploads image to GCP |
| Deploy | `gcloud run deploy polyp-app ...` | Deploys container to Cloud Run |

---
