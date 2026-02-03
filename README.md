# project_rl_Lab

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/phc_isaaclab
    ```
```bash
python -m pip install -e source/phc_isaaclab

python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-v0 --headless

python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-gym-v0 --headless

python scripts/rsl_rl/play.py --task=Unitree-G1-29dof-Velocity-gym-v0
```