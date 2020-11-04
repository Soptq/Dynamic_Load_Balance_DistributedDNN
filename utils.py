def print_layer(model, layer_name):
    for name, param in model.named_parameters():
        if name == layer_name:
            return param


def get_batch(source, i, bptt):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target