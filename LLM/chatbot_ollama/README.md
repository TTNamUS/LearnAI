# Mini-project


##

```bash

```

### Step 1:

### Step 2:  
Install the requirements:
```bash
pip install -r requirements.txt
```


### Step 3. Running Ollama with Docker

Docker allows you to run applications in isolated environments called containers. To run Ollama with Docker, follow these steps:

1. **Pull the Ollama Docker Image**: Open your terminal and run the following command to pull the Ollama Docker image from Docker Hub:

    ```bash
    docker pull ollama/ollama
    ```

2. **Run the Ollama Container**: Execute the following command to run the Ollama container:

    ```bash
    docker run -it \
        --rm \
        -v ollama:/root/.ollama \
        -p 11434:11434 \
        --name ollama \
        ollama/ollama
    ```

    > **Check the Ollama Version**: To find the version of the Ollama client, enter the container and execute:
    >```bash
    > docker exec -it ollama ollama -v
    > ```

3. **Downloading the LLM model**  
To download a large language model (LLM) using Ollama, follow these steps:

- **Enter the Container**   
If you are not already inside the container, enter it using:

    ```bash
    docker exec -it ollama bash
    ```

- **Download the Model**  
Run the following command inside the Docker container to download the `phi3` model:

    ```bash
    ollama pull phi3
    ```

    > **Run the LLM model inside the container**
    > ```bash
    > ollama run phi3
    > ```


[!NOTE]:  
To avoid downloading the model weights every time you run the container, follow these steps:

4.  **Downloading the Weights**
    - **Create a Local Directory**: Create a local directory to store the model weights:

    ```bash
    mkdir ollama_files
    ```

    - **Run the Container with Volume Mapping**: Run the container with the local directory mapped to the container's `/root/.ollama` directory:

    ```bash
    docker run -it --rm -v ./ollama_files:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```

    - **Download the Model**: Enter the container and download the model:

    ```bash
    docker exec -it ollama bash
    ollama pull phi3
    ```


5. **Adding the Weights**  
To create a new Docker image with the downloaded weights, follow these steps:

    - **Create a Dockerfile**: Create a file named `Dockerfile` (without any extension) with the following content:

    ```dockerfile
    FROM ollama/ollama
    COPY ./ollama_files /root/.ollama
    ```

    - **Build the Docker Image**: Build the new Docker image:

    ```bash
    docker build -t ollama-phi3 .
    ```

6. **Run the New Docker Image**: Run the new Docker image:
    ```bash
    docker run -it --rm -p 11434:11434 ollama-phi3
    ```



### Step 4: 

