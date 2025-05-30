# Base image: NVIDIA CUDA 12.4.1 with cuDNN (devel version for toolkit flexibility, on Debian 12)
FROM nvidia/cuda:12.4.1-devel-debian12

# --- Environment Setup ---
# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive
# Set a default timezone
ENV TZ=America/Los_Angeles
# Ensure Python output is sent straight to terminal (Docker logs) without being buffered
ENV PYTHONUNBUFFERED=1
# Add user's local bin to PATH (for pip installed executables) ??? Not sure path
ENV PATH="/root/.local/bin:${PATH}"

# --- Install System Dependencies ---
# Update package lists and install Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    openssh-client \ 
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# --- Configure Python ---
# Make python3.11 the default 'python' and ensure pip points to python3's pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    # Use python3.11 to ensure pip is for this version
    python3.11 -m ensurepip --upgrade

# Upgrade pip, setuptools, and wheel for the selected Python version
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Application Setup ---
# Set the working directory inside the container
WORKDIR /app/fine_tuning
# Copy the requirements file first to leverage Docker layer caching.
COPY requirements.txt .

# Install Python dependencies from your requirements file.
# This will install torch, transformers, accelerate, peft, datasets,
# and all the nvidia-*-cu12 packages, setting up the CUDA environment
# for PyTorch from within the Python ecosystem.
RUN echo "Starting pip install from requirements.txt..." && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    echo "Finished pip install."

# Copy the rest of your fine-tuning application code into the container.
# This includes your fine_tune_script.py and any other necessary Python modules,
# configuration files, or utility scripts located in the build context (fine_tuning/ directory).
COPY . .

# --- Default Command ---
# Keep the container running. This is useful for:
# 1. VS Code Dev Containers: VS Code can attach to this running container.
# 2. Manual `docker exec`: You can start the container and then `docker exec` into it to run scripts or commands.
# 3. Interactive sessions: `docker run -it ... /bin/bash`
# If you have a primary script you always run, you could change this to:
# ENTRYPOINT ["python", "your_main_fine_tuning_script.py"]
# CMD ["--default-arg1", "value1"] # Default arguments for your script
CMD ["sleep", "infinity"]