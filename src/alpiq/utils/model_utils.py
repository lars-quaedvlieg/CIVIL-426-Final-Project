from prettytable import PrettyTable


def print_num_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        if parameter.requires_grad:
            total_trainable_params += params
    print(table)
    print(f"Total Params: {total_params}")
    print(f"Total Trainable Params: {total_trainable_params}")
    return total_params
