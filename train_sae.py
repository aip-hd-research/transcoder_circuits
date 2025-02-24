# SAE training

"""
This sample script can be used to train a transcoder on a model of your choice.
This code, along with the transcoder training code more generally, was largely
    adapted from an older version of Joseph Bloom's SAE training repo, the latest
    version of which can be found at https://github.com/jbloomAus/SAELens.
Most of the parameters given here are the same as the SAE training parameters
    listed at https://jbloomaus.github.io/SAELens/training_saes/.
Transcoder-specific parameters are marked as such in comments.

"""
import argparse
from dataclasses import asdict

import torch
import os
import sys
import numpy as np
import wandb
from wandb.cli.cli import offline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model

def main(args):
    lr = 4e-4  # learning rate
    l1_coeff = 5  # l1 sparsity regularization coefficient
    expansion_factor = 8

    batch_size = 4096
    per_device_batch_size = None
    total_training_tokens = 50_000_000
    l1_warm_up_steps = 5000

    cfg = LanguageModelSAERunnerConfig(
        hook_point="blocks.17.ln2.hook_normalized",
        hook_point_layer=17,
        d_in=1600,
        dataset_path="Skylion007/openwebtext",
        is_dataset_tokenized=False,

        # using on of 'CodeLlama-7b-Instruct-hf', 'MistralHermes-CodePro-7B-v1' and 'Magicoder-S-DS-6.7B'
        model_name="gpt2-xl",

        # SAE Parameters
        expansion_factor=expansion_factor,
        b_dec_init_method="mean", # can also use zeros

        # Training Parameters
        lr=lr,
        l1_coefficient=l1_coeff,
        lr_scheduler_name="constantwithwarmup",
        train_batch_size=batch_size,
        per_device_batch_size=per_device_batch_size,
        context_size=128,  # will control the lenght of the prompts we feed to the model. Larger is better but slower.
        lr_warm_up_steps=l1_warm_up_steps,

        # Activation Store Parameters
        n_batches_in_buffer=32,  # Must be large enough so that n_batches_in_buffer * store_batch_size * context_size >= 2 * train_batch_size
        total_training_tokens=total_training_tokens,
        store_batch_size=4,
        data_column="text",
        improve_mixing=True,  # Disabling this mean you can get as low as n_batches_in_buffer * store_batch_size * context_size >= train_batch_size

        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_method=None,
        feature_sampling_window=1000,
        resample_batches=1028,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        top_k=None,

        # WANDB
        log_to_wandb=True,
        wandb_project="SAE_gpt2-xl",
        wandb_entity="pvs-shared",
        wandb_group=None,
        wandb_log_frequency=10,

        # Misc
        use_tqdm=True,
        device="cuda:0",
        seed=42,
        n_checkpoints=0,
        checkpoint_path="/nfs/data/shared/gpt2-xl-saes",  # change as you please
        dtype=torch.float32,
        model_dtype=torch.float16,
        model_device="cuda:0",
        lazy_device_loading=False,
    )


    print(f"About to start training with lr {lr} and l1 {l1_coeff}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(cfg)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, group=cfg.wandb_group, config=asdict(cfg),
                   mode="offline" if args.offline else "online")

    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    # train SAE
    sparse_autoencoder = train_sae_on_language_model(cfg,
        model, sparse_autoencoder, activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_method=cfg.feature_sampling_method,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency
    )

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=False)

    main(parser.parse_args())
