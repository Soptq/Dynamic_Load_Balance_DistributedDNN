def print_layer(model, layer_name):
    for name, param in model.named_parameters():
        if name == layer_name:
            return param