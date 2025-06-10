### Initial setup guide

This guide assumes a correctly configured Linux machine with a GPU. All the following actions should be performed in a terminal of your choosing.

#### Step 1: `conda`

We will need an isolated environment, so let's check if we have `conda` installed:

```bash
which conda
```

If you get an output from this command other than `conda not found`, then you can proceed to the next step. Otherwise, install a `miniconda` package by running following set of commands one by one:

```bash
mkdir -p ~/miniconda3 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all
```

Now close and reopen the terminal application so that `conda` becomes available.

#### Step 2: Creating environment

Now let's create a python environment that we will work in:

```bash
conda create -n reasoning_uq python==3.10
```

After it finishes installing a 3.10 version of python into the environment called `reasoning_uq` we can activate it with the following command:

```bash
conda activate reasoning_uq
```

You should be able to see the name of the environment near your shell prompt. This way you can always see which environment you are currently working in.

#### Step 3: Installing packages

The main library that you will work with on this project is called `LM-Polygraph`. It's a collection of baseline methods for UQ in LLMs and an easy-to-run benchmark allowing you to compare how different UQ method fare against each other. Since we will need to modify the code of `LM-Polygraph` and possibly add new methods of uncertainty quantification, we will need to install it in _editable_ mode. 

To do that let's first create a directory where all our code will be stored:

```bash
mkdir -p ~/workspace
cd ~/workspace
```

Since we will need to work on `LM-Polygraph` code, let's install it directly from Github:

```bash
git clone git@github.com:IINemo/lm-polygraph.git
```

Now we can install it:

```bash
cd lm-polygraph
pip install -e .
```

When this finishes, we can check if everything is installed correctly. Let's see if the main executable script of the library is available

```bash
which polygraph_eval
```

This should output a path to the script somewhere in the conda working directory. Let's also make sure that library is available from the python interpreter. Start the python REPL environment like this:

```
python
```

And try running the following command in it:

```python
from lm_polygraph.utils.manager import UEManager
```

If this succeeds without any errors - you have installed `LM-Polygraph` correctly. Last thing to check is whether GPU is available on your machine. Run

```python
import torch
torch.cuda.is_available()
```

which should evaluate to `True`.

You now have everything set up correctly to start working on this project.
