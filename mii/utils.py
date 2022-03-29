'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import sys
import os
import logging
import importlib

from mii.constants import MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT, MII_DEBUG_MODE, \
    MII_DEBUG_MODE_DEFAULT, MII_DEBUG_DEPLOY_KEY, MII_DEBUG_BRANCH, MII_DEBUG_BRANCH_DEFAULT


def get_model_path():
    aml_model_dir = os.getenv('AZUREML_MODEL_DIR')
    if aml_model_dir is not None:
        return aml_model_dir

    mii_model_dir = os.getenv('MII_MODEL_DIR')

    if mii_model_dir is not None:
        return mii_model_dir

    #assert False, "MII_MODEL_DIR must be set if not running on AML. Current value is None"
    #TODO remove this and uncomment above line. Only doing this for debugging
    return "temp_model"


def is_aml():
    return os.getenv('AZUREML_MODEL_DIR') is not None


def set_model_path(model_path):
    os.environ['MII_MODEL_DIR'] = model_path


def mii_cache_path():
    cache_path = os.environ.get(MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
    return cache_path


def import_score_file():
    spec = importlib.util.spec_from_file_location(
        'score',
        os.path.join(mii_cache_path(),
                     "score.py"))
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score


def generated_score_path():
    return os.path.join(mii_cache_path(), "score.py")


def debug_score_preamble():
    preamble = ""
    debug_mode_enabled = int(os.environ.get(MII_DEBUG_MODE, MII_DEBUG_MODE_DEFAULT))
    if not debug_mode_enabled:
        return preamble

    deploy_key = os.environ.get(MII_DEBUG_DEPLOY_KEY)
    debug_branch = os.environ.get(MII_DEBUG_BRANCH, MII_DEBUG_BRANCH_DEFAULT)
    key_path = "/tmp/mii_deploy_key"

    preamble = f"""
import subprocess, os, sys
deploy_key = '''{deploy_key}'''
with open('{key_path}', 'w') as fd:
    fd.write(deploy_key)
    fd.write("\\n")
subprocess.run(['chmod', '600', '{key_path}'])
env = os.environ.copy()
env["GIT_SSH_COMMAND"]="ssh -i {key_path} -o StrictHostKeyChecking=no"
install_cmd = "-m pip install git+ssh://git@github.com/microsoft/DeepSpeed-MII.git@{debug_branch}"
subprocess.run([sys.executable] + install_cmd.split(" "), env=env)
"""
    return preamble


def setup_task():
    return get_model_path(), not is_aml(), is_aml()


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="MII", level=logging.INFO)