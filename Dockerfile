# Use official Python lightweight image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the ADK CLI is installed
RUN pip install google-adk==1.14.0

# Copy the rest of your agent code
COPY . .

# Expose the port the UI will run on
EXPOSE 8000

# Start the ADK local server with UI attached, bound to 0.0.0.0 so Docker exposes it
CMD ["adk", "run", ".", "--with_ui", "--host", "0.0.0.0", "--port", "8000"]