# How to use this project

This guide explains how to create a Python virtual environment (venv), install Jupyter inside it, register it as a kernel, and launch Jupyter Notebook using that environment.
The project is then made to be used through Viz.ipynb 
---

## 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install the dependencies

```bash
pip install -r requirements.txt
```

## 3. Create a jupyter kernel linked to this venv 

```bash
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```


## 4. Start a notebook

```bash
jupyter notebook
```
