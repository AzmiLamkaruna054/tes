# 1. Use a base image with Python
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (optional, jika diperlukan)
RUN apt-get update && apt-get install -y build-essential

# 4. Copy the application code
COPY . /app/

# 5. Create virtual environment and install dependencies
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install -r requirements.txt

# 6. Update PATH to use the virtual environment's Python and pip
ENV PATH="/opt/venv/bin:$PATH"

# 7. Set the command to run the app
CMD ["gunicorn", "app:app"]
