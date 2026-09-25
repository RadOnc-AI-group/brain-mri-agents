
The Docker image of SynthSeg will be pulled from `ghcr.io` at the project setup.
If the pull doesn't work or the image is broken, please build it yourself by following these steps:

1. Clone original SynthSeg repository to your home directory:

    ```bash                                   
    cd ~                           
    git clone https://github.com/BBillot/SynthSeg.git
    ``` 


2. Copy the Dockerfile and requirements from this directory into `~/SynthSeg/`:

    ```bash
    cp requirements_python3.11.txt ~/SynthSeg/
    cp SynthSeg.Dockerfile.python3.11 ~/SynthSeg/
    ```

3. Navigate to `~/SynthSeg` and build the Docker image:

    ```bash
    cd ~/SynthSeg
    docker build -t synthseg-robust-new -f SynthSeg.Dockerfile.python3.11 .
    ```