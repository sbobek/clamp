# Cluster Analysis with Multidimensional Prototypes (CLAMP)
## About
## Instalation
First of all clonde the repository and its submodules, and enter it:

```
git clone https://github.com/sbobek/clamp.git
cd clamp
git submodule update --init --recursive
```
Some of the packages used in CLAMP anre not available in conda, hence the following code should set up all of the requirements in virtual environment:

```
conda create --name clampenv python=3.8
conda activate clampenv
conda install pip
pip install -r requirements.txt
```

Additionally if you want to wotk with [JupyterLab](https://jupyter.org/) install it and raun it, while being in active `clampenv` envoronment by:

```
pip install jupyter lab
jupyter lab
```

Open `usage_example.ipynb` in your JupyterLab and see how CLAMP works in practice.

## Usage example
## Cite this work
TBA
