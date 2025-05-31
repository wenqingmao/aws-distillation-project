# aws-distillation-project
Lightweight Medical QA System: Knowledge Distillation on PubMedQA using Transformers and Domain-Specific Models

---

## How to Run the Application

This section describes how to set up and run the PubMedQA Analysis application on the target Google Compute Engine (GCE) VM. The application consists of a FastAPI backend serving a distilled student model (using GPU) and a Streamlit frontend for user interaction.

Before you begin, ensure the following are met:

1.  **GCE VM Access:**
    * SSH access to the designated GCE VM.
    * The VM must have an NVIDIA GPU attached (e.g., NVIDIA L4) with appropriate NVIDIA drivers installed.
    * The VM should have a **Static Public IP Address**.

2.  **Software on GCE VM:**
    * **Git:** For cloning the project repository.
    * **Docker Engine:** Installed and running. (See [official Docker installation guide](https://docs.docker.com/engine/install/))
    * **Docker Compose:** Installed (usually comes as a Docker plugin, e.g., `docker compose`). (See [official Docker Compose installation guide](https://docs.docker.com/compose/install/))
    * **NVIDIA Container Toolkit:** Installed and configured to allow Docker containers to access the GPU. (See [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

3.  **Model Files:**
    * The distilled student model files (e.g., `config.json`, `model.safetensors`, `tokenizer.json`, etc.) must be present on the GCE VM at the following absolute path:
        `/home/shared/models/student_distilled/`

4.  **Source Code:**
    * You will need the application's source code (the `app` directory containing `docker-compose.yml`, `backend/`, and `frontend/` subdirectories).

5.  **GCP Firewall Rules:**
    * A GCP firewall rule must be configured to allow **Ingress TCP traffic on port `8501`** from source IP range `0.0.0.0/0` to your GCE VM (or VMs with the appropriate target tag). This port is used for direct access to the Streamlit application.

### Setup

1.  **Get the Source Code:**
    * **If using Git:** Clone the repository to your GCE VM.
        ```bash
        git clone git@github.com:wenqingmao/aws-distillation-project.git
        ```

2.  **Navigate to the Project Directory:**
    All subsequent commands should be run from the directory containing your `docker-compose.yml` file (this is typically the `app/` directory).
    ```bash
    cd aws-distillation-project/app
    ```

3.  **Verify Model Files:**
    Double-check that the student model files are indeed present at `/home/shared/models/student_distilled/` on the VM, as the backend service relies on mounting this path.

### Running the Application

1.  **Build and Start Docker Containers:**
    This command will build the Docker images for the backend and frontend and then start the services in detached mode.
    ```bash
    sudo docker compose up --build -d
    ```
    If no code, Dockerfiles, or requirements have changed since your last successful build, you can simply run the following command to start the container:
    ```bash
    sudo docker compose up -d
    ```
2.  **Check Service Status:**
    Allow a few moments for the containers to start, especially the backend which needs to load the ML model. You can check the status of your services:
    ```bash
    sudo docker compose ps
    ```

### Accessing the Application

Once the services are up and running:

* Open your web browser and navigate to:
    **`http://<YOUR_VM_PUBLIC_IP>:8501`**
    (Replace `<YOUR_VM_PUBLIC_IP>` with the actual static public IP address of your GCE VM.)

### Stopping the Application

To stop and remove the running containers, networks, and default volumes created by Docker Compose:

1.  **Navigate to the Project Directory:**
    ```bash
    cd /aws-distillation-project/app
    ```
2.  **Run Docker Compose Down:**
    ```bash
    sudo docker compose down
    ```
    This command will stop and remove the containers. Your Docker images will remain on the system, as will your model files at `/home/shared/models/student_distilled/`.

