from setuptools import setup

setup(
    name='transcoder_circuits',
    version='0.2.2',
    packages=['huggingface', 'sae_training', 'sae_training.geom_median', 'transcoder_circuits'],
    url='https://github.com/aip-hd-research/transcoder_circuits',
    license='MIT',
    author='Jacob Dunefsky, and AIP members',
    author_email='artur.andrzejak@uni-heidelberg.de',
    description='Adaptation of transcoder_circuits to CodeLlama',
    # The following requirements are incomplete, to simplify using it in e.g. my-rome
    install_requires = [
        # "matplotlib",
        # "numpy",
        # "plotly",
        # "tqdm",
        # "datasets",
        # "fsspec",
        # "einops",
        # "setuptools",
        # "wandb",
        # "huggingface_hub",
        # "jupyter",
        # "notebook",
        # "pip"
    ]

)
