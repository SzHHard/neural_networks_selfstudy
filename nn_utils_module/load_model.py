import torch
from pathlib import Path
from neural_networks_self_study.nn_utils_module.test_nn import test_nn_accuracy_only

device = torch.accelerator.current_accelerator().type

def load_latest_model(name_prefix, model_dir="../saved_models"):
    save_dir = Path(model_dir)
    model_files = list(save_dir.glob(pattern=f"{name_prefix}*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No models found in {save_dir} prefixed by {name_prefix}")

    # Выбираем самый новый файл
    latest_model_path = max(model_files, key=lambda x: x.stat().st_ctime)
    return torch.jit.load(latest_model_path)



