# Use the official Python image
FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /app

# Copy all application files
COPY . .

# Grant ownership of the app directory to the user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Install dependencies
RUN pip install --user -r requirements.txt

# Add the user's local bin directory to the PATH
ENV PATH="/home/user/.local/bin:${PATH}"

# Set the entrypoint to the Streamlit app
CMD ["streamlit", "run", "app.py"]
