import logging

from GAN import GAN
from WGAN import WGAN
from utils import init_working_space

# from WGANGP import WGANGP


if __name__ == "__main__":
    # TODO: Change these to config_file
    method = 'wgan'  # method name can be 'gan' or 'wgan' or 'wgan-gp'
    opt = 'adam_beta'  # Optimizer can be 'adam_beta' or 'rmsprop' or 'adam'

    RUN_FOLDER = init_working_space(method)
    model = None
    if method == 'gan':
        model = GAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan':
        model = WGAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan-gp':
        # model = WGAN_GP(path=RUN_FOLDER, optimizer=opt, visualize=False)
        pass
    else:
        logging.critical('Unknown Method Name, you can choose either gan, wgan, wgan-gp')
        exit(0)

    model.save()
    model.plot_visualize_model()
    model.train()
