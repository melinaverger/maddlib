import sys

def activate_myvenv():
    path_to_venv = "../../virtual_environments/venv_for_maddlib/lib/python3.10/site-packages"
    sys.path.append(path_to_venv)
