# Use the official Python image as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# # Expose port 80 for the application, 7860 for Gradio UI
EXPOSE 8000


# Start the application
# docker build -t <container-name> .
# docker run -p 7860:7860 -p 8000:8000 <container-name>
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]
