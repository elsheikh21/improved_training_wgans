from utils import init_working_space
from GAN import GAN


if __name__ == "__main__":
    # TODO: Change these to config_file
    method = 'gan'  # method name can be 'gan' or 'wgan' or 'wgan-gp'
    opt = 'adam_beta'  # Optimizer can be 'adam_beta' or 'rmsprop' or 'adam'

    RUN_FOLDER = init_working_space(method)
    gan = GAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    gan.save()
    gan.plot_visualize_model()
    gan.train()
