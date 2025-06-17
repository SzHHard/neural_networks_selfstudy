from neural_networks_self_study.nn_utils_module.load_model  import load_latest_model
from neural_networks_self_study.nn_utils_module.test_nn  import test_nn_accuracy_only
from data_loaders import get_data_loaders

_, test_loader = get_data_loaders(64)

def test_latest_model_by_prefix(name_prefix, loader):
    model = load_latest_model(name_prefix)
    model.eval()
    print(f"Testing {name_prefix}...")
    test_nn_accuracy_only(model, loader)

test_latest_model_by_prefix('cifar10_residual_connections', test_loader)

test_latest_model_by_prefix('cifar10_simple_cnn', test_loader)