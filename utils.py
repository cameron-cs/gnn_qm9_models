import torch
import logging
import yaml


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logger(log_file):
    logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    return logger
