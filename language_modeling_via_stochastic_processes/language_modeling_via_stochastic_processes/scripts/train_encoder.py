import os
import sys
import wandb
import getpass
import random, torch, numpy
from language_modeling_via_stochastic_processes.src.systems import (
    brownian_bridge_system,
    cbow_system,
    cbow_diff_system,
    cbow_diff_genslm_system,
    cbow_dot_system,
    cbow2_system,
    brownian_system,
    vae_system,
    infonce_system
    )

import pytorch_lightning as pl
from language_modeling_via_stochastic_processes.src.evaluation import recovery
import hydra

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'brownian_bridge': brownian_bridge_system.BrownianBridgeSystem,
    'cbow': cbow_system.CBowSystem,
    'cbow_diff': cbow_diff_system.CBowSystem,
    'cbow_diff_genslm': cbow_diff_genslm_system.CBowSystem,
    'cbow_dot': cbow_dot_system.CBowSystem,
    'cbow2': cbow2_system.CBow2System,
    'brownian': brownian_system.BrownianSystem,
    'vae': vae_system.VAESystem,
    'infonce': infonce_system.InfoNCESystem
}

#@hydra.main(config_path="../config/encoder", config_name="brownian_bridge")
@hydra.main(config_path="../config/encoder", config_name="cbow")
def run(config):
    print("Running in online mode")
    os.environ['WANDB_MODE'] = 'online'
    # if config.wandb_settings.dryrun:
    #     print("Running in dryrun mode")
    #     os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'
    print (f'seed: {config.experiment_params.seed}')

    seed_everything(
        config.experiment_params.seed,
        use_cuda=config.experiment_params.cuda)

    wandb.init(
        project=config.wandb_settings.project,
        name=config.wandb_settings.exp_name,
        group=config.wandb_settings.group,
        config=config,
    )

    print("CKPT AT {}".format(
        os.path.realpath(os.path.join(config.wandb_settings.exp_dir,
        'checkpoints'))))
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath = os.path.join(config.wandb_settings.exp_dir, 'checkpoints'),
        save_last=False,
        save_top_k=1,
        monitor="val_loss",
        mode='min',
    )

    SystemClass = SYSTEM[config.loss_params.loss]
    print (config.loss_params.loss)
    print (SystemClass)
    system = SystemClass(config)

    trainer = pl.Trainer(
        default_root_dir=config.wandb_settings.exp_dir,
        gpus=1,
        #checkpoint_callback=ckpt_callback,
        callbacks=[ckpt_callback],
        max_epochs=int(config.experiment_params.num_epochs),
        min_epochs=int(config.experiment_params.num_epochs),
        fast_dev_run=False,
    )

    trainer.fit(system)

    ## Save the model
    #system.save(directory=wandb.run.dir)

    ## Evaluation:
    #trainer.test(system)
    trainer.test(system, ckpt_path="best")

def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    sys.argv.append(f'hydra.run.dir={os.getcwd()}')
    run()



