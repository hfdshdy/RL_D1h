 tensorboard   --logdir /root/RL_D1/RL_D1h/logs/co_rl/d1h_flat_velocity/ppo/2026-05-06_19-55-13   --host 0.0.0.0   --port 6006
看板
tensorboard \
  --logdir /root/RL_D1/RL_D1h/logs/co_rl/d1h_ddt_flat_velocity/ddt_ppo \
  --host 0.0.0.0 \
  --port 6006 \
  --reload_interval 5



训练ddt-flat
python scripts/co_rl/ddt_train.py \
  --task Isaac-Velocity-Flat-D1h-DDT-v0 \
  --headless \
  --enable_cameras \
  --num_envs 2048 \
  --video \
  --video_length 200 \
  --video_interval 5000 \
  --max_iterations 1500 \
  --resume True \
  --load_run 2026-05-14_20-27-46 \
  --checkpoint model_7950.pt 



训练
python scripts/co_rl/train.py \
  --task Isaac-Velocity-Flat-D1h-v0 \
  --algo ppo \
  --headless \
  --enable_cameras \
  --num_envs 2048 \
  --video \
  --video_length 200 \
  --video_interval 5000 \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --resume True \
  --load_run 2026-05-09_17-03-16 \
  --checkpoint model_9450.pt \
  --max_iterations 1000


推理
  python scripts/co_rl/play.py \
  --task Isaac-Velocity-Flat-D1h-Play-v0 \
  --algo ppo \
  --headless \
  --enable_cameras \
  --num_envs 6 \
  --video \
  --video_length 2000 \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --load_run 2026-05-09_21-45-27 \
  --checkpoint model_9850.pt 


  键盘演示
W 或上方向键：前进
S 或下方向键：后退
A 或左方向键：左转
D 或右方向键：右转
Q：抬高
E：降低
Space：速度清零
R：高度归零
Esc：速度清零并高度归零
  python scripts/co_rl/keyboard_play.py \
  --task Isaac-Velocity-Flat-D1h-Play-v0 \
  --algo ppo \
  --num_envs 6 \
  --video \
  --video_length 2000 \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --load_run 2026-05-08_16-51-53 \
  --checkpoint model_6950.pt 


# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/RL_D1h

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/RL_D1h/RL_D1h/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/RL_D1h"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
