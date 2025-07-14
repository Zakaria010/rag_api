# Use a Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# (OPTIONAL) If you have a requirements.txt, you can use this instead:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install your Python dependencies
RUN pip install fastapi uvicorn faiss-cpu

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api_rag:app", "--host", "0.0.0.0", "--port", "8000"]
