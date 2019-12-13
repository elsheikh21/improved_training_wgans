import logging

from models import GAN, WGAN, WGAN_GP
from utils import init_working_space, load_config

if __name__ == "__main__":
    config_params = load_config()
    method = config_params.get('method')
    opt = config_params.get('optimizer')
    RUN_FOLDER = init_working_space(method)

    model = None
    if method == 'gan':
        model = GAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan':
        model = WGAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan-gp':
        model = WGAN_GP(path=RUN_FOLDER, optimizer=opt, visualize=False)
    else:
        logging.critical('Unknown Method Name, you can choose either gan, wgan, wgan-gp')
        exit(0)

    model.save()
    model.plot_visualize_model()
    model.train()
