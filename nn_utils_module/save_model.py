import torch
import os
from pathlib import Path
import datetime

def save_model(model, filename):
    save_dir = Path("../saved_models")
    save_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = save_dir / f"{filename}_{timestamp}.pt"
    scripted_model = torch.jit.script(model)
    scripted_model.save(path)
    print(f"Веса модели сохранена в: {path.absolute()}")


