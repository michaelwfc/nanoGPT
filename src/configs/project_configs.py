import tempfile
import logging
import random
import config
import os
import yaml
import logging
logger = logging.getLogger(__name__)


def initialize_config_and_logging(existing_config=None):
    global global_config
    global_config = build_config(existing_config)
    setup_logging(global_config)
    logger.debug("Config: " + str(yaml.dump(global_config.as_dict())))
    return global_config


def get_config():
    global global_config
    assert global_config is not None
    return global_config


def build_config(existing_config=None):
    configs = [
        # Using config library
        config.config_from_env(prefix="LLAMA", separator="_", lowercase_keys=True),
    ]

    if existing_config:
        if isinstance(existing_config, dict):
            configs.append(config.config_from_dict(existing_config))
        else:
            configs.append(existing_config)

    config_paths = get_config_paths()

    for path in reversed(config_paths):
        print("Loading builtin config from " + path)
        configs.append(config.config_from_yaml(path, read_from_file=True))

    return config.ConfigurationSet(*configs)


def get_config_paths():
    paths = []


def get_config_paths():
    paths = []

    config_name = "llama_config"
    config_base = "configs"

    base_config_path = os.path.join(config_base, config_name + ".yaml")
    if os.path.exists(base_config_path):
        paths.append(base_config_path)

    local_config_path = os.path.join(config_base, config_name + "_local.yaml")
    if os.path.exists(local_config_path):
        paths.append(local_config_path)

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, "." + config_name + ".yaml")
    if os.path.exists(home_config_path):
        paths.append(home_config_path)

    return paths


def setup_logging(arguments):
    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("smart_open").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
