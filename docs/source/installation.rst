.. _installation:

Installation
============

To install e3k, follow these steps:

For conda users
---------------
1. Download the `env.yaml` file from this project's Github repository.

2. Navigate to the dowloaded file. 

3. Create the environment using the `env.yaml` file:
   
   .. code-block:: shell
    
        conda env create -f env.yaml

4. Activate the newly created environment:

   .. code-block:: shell
    
        conda activate <env_name>

5. Install the wheel package using pip:

   .. code-block:: shell
    
        pip install wheel

For non-conda users 
----------------------
For non-conda users having python 3.10 is necessary. 

1. Download the `requirements.txt` file from this project's Github repository.

2. Navigate to the dowloaded file. 

3. Install the required packages using pip:

   .. code-block:: shell
    
        pip install -r requirements.txt

4. Install the wheel package using pip:

   .. code-block:: shell
    
        pip install wheel

For poetry users
-----------------
1. Reconsider your life choices.

Verify the installation:
------------------------

   .. code-block:: shell

       e3k --version