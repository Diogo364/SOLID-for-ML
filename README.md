# SOLID For Machine Learning
## Description
TODO

## Setting Environment
For setting the environment, one might use one of the following methods:

### 1. Make
It is possible to use `Make` to set the entire environment. 
1. Open the `Makefile` file in your editor and update the `SHELL` variable acording to your system.
2. run the `setup` command from `Make`:
    ```
    make setup
    ```
3. Now you're good to go!

### 2. Conda
It is possible to create the conda environment by following the following steps:
1. Create the conda env using the `environment.yml` file
    ```
    conda env create -f environment.yml
    ```
2. Activate the the `solid-ml` env
   ```
    conda activate solid-m
   ```
3. follow the steps in the Download Data section.

### 3. Venv 
It is also possible to use `Python's` `venv` following the following steps:
1. Create the `venv`
    ```
    python -m venv solid-ml
    ```
2. Activate the the `solid-ml` env
   ```
    # This command works for linux. Refeer to venv documentation
    source solid-ml/bin/activate
   ```
3. follow the steps in the Download Data section.

## Download Data
To get the data, you must execute the `get_wine_data.py` python script from the `setup` folder.
```
python setup/get_wine_data.py
```
> **IMPORTANT:** If you used Make to setup the environment you don't need to run this step

## Requirements
- `conda==4.13.0`
- `python==3.10`