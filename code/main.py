from models import GAN, WGAN, WGANGP
from utils import init_working_space, load_config

if __name__ == "__main__":
    config_params = load_config()
    method = str(config_params.get('method')).lower()
    assert method in ['gan', 'wgan', 'wgan-gp'], 'Unknown Method Name, you can choose either gan or wgan or wgan-gp'

    opt = str(config_params.get('optimizer')).lower()
    assert opt in ['adam_beta', 'adam',
                   'wgan'], "Unknown Optimizer Name, you can choose either adam_beta or adam or rmsprop"

    RUN_FOLDER = init_working_space(method)

    model = None
    if method == 'gan':
        model = GAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan':
        model = WGAN(path=RUN_FOLDER, optimizer=opt, visualize=False)
    elif method == 'wgan-gp':
        model = WGANGP(path=RUN_FOLDER, optimizer=opt, visualize=False)

    assert model is not None, 'Method name is incorrect'
    model.save()
    model.plot_visualize_model()
    model.train()
