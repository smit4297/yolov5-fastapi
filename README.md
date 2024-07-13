Here's the revised README.md file with some improvements for clarity and consistency:

```markdown
# YOLOv5 FastAPI Project

This project implements a FastAPI application using YOLOv5 for object detection.

## Setup Without Docker

### Prerequisites
- Python 3.9
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/smit4297/yolov5-fastapi.git
   cd yolov5-fastapi
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI application:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

5. The API should now be accessible at `http://localhost:8000`

6. Open `yolo_object_detection.html` in your browser.

## Setup With Docker

### Prerequisites
- Docker

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/smit4297/yolov5-fastapi.git
   cd yolov5-fastapi
   ```

2. Build the Docker image:
   ```bash
   docker build -t yolov5-fastapi .
   ```

3. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 yolov5-fastapi
   ```

4. The API should now be accessible at `http://localhost:8000`

## Environment Variables

The Docker setup uses the following environment variables:
- `WORKERS_PER_CORE`: Set to 4
- `MAX_WORKERS`: Set to 24
- `LOG_LEVEL`: Set to "warning"
- `TIMEOUT`: Set to 200

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Dockerfile

```dockerfile
FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="team-erc"

ENV WORKERS_PER_CORE=4
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT=200

RUN mkdir /yolov5-fastapi

COPY requirements.txt /yolov5-fastapi
COPY . /yolov5-fastapi

WORKDIR /yolov5-fastapi

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Maintainer

This project is maintained by team-erc.
```

This README provides comprehensive instructions for setting up the project both with and without Docker, includes information about environment variables and API documentation, and incorporates the Dockerfile content.
