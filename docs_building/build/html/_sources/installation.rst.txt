.. _installation:

Installation
============

To install e3k, follow these steps:

Using Dockerfile
-----------------

   .. code-block:: shell 
      
      docker pull jamwojt/cli:1
      docker run -it --rm jamwojt/cli:1

For conda users
---------------
1. Download the `environment.yaml` file from this project's Github repository (environment folder).

2. Navigate to the dowloaded file. 

3. Create the environment using the `environment.yaml` file:
   
   .. code-block:: shell
    
        conda env create -f environment.yaml

4. Activate the newly created environment:

   .. code-block:: shell
    
        conda activate <env_name>

5. Install the wheel package using pip:

   .. code-block:: shell
    
        pip install <wheel_filename>

For non-conda users 
----------------------
For non-conda users having python 3.8 is necessary. 

1. Download the `requirements.txt` file from this project's Github repository (environment folder).

2. Navigate to the dowloaded file. 

3. Install the required packages using pip:

   .. code-block:: shell
    
        pip install -r requirements.txt

4. Install the wheel package using pip:

   .. code-block:: shell
    
        pip install <wheel_filename>

Verify the installation:
------------------------

   .. code-block:: shell

       e3k --version