FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Create output and logs directories
RUN mkdir -p output logs data

# Install the local package
RUN pip install -e .

# Set the entry point
ENTRYPOINT ["python", "run.py"]

# Default command if no arguments are provided
CMD ["--help"]