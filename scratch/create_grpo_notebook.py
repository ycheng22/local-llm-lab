import json
import os

notebook_path = r'../notebooks/grpo_training.ipynb'

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 3: GRPO / RLVR Training\n",
    "This notebook runs the Reinforcement Learning pipeline using `configs/grpo_v1_5k.yaml`.\n",
    "\n",
    "We invoke `grpo.py` natively through the Python kernel to ensure progress bars, VRAM statistics, and loss logs stream live without buffering issues."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import torch\n",
    "\n",
    "# Clear VRAM before starting\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "sys.path.append(os.path.abspath(\"../src\"))\n",
    "sys.argv = [\"grpo.py\", \"--config\", \"../configs/grpo_v1_5k.yaml\"]\n",
    "\n",
    "from llm_lab.training.grpo import main\n",
    "main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Created {notebook_path}")
