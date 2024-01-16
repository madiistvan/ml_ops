import hydra

hydra.initialize(config_path="config", version_base=None)
train_config = hydra.compose(config_name="train_config")
