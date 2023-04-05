'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from typing import Optional, Dict, Any
import json

from deepspeed.launcher.runner import fetch_hostfile

import mii
from .constants import DeploymentType
from .utils import logger
from .models.score import create_score_file
from .config import ReplicaConfig, LoadBalancerConfig


def deploy(task: str,
           model: str,
           deployment_name: str,
           deployment_type: str = DeploymentType.LOCAL,
           model_path: Optional[str]=None,
           enable_deepspeed: bool = True,
           enable_zero: bool = False,
           ds_config: Dict[str,Any]=None,
           mii_config: Optional[str, Dict, None]={},
           version: int =1,
           **kwargs):
    """Deploy a task using specified model. For usage examples see:

        mii/examples/local/text-generation-example.py


    Arguments:
        task: Name of the machine learning task to be deployed.Currently MII supports the following list of tasks
            ``['text-generation', 'text-classification', 'question-answering', 'fill-mask', 'token-classification', 'conversational', 'text-to-image']``

        model: Name of a supported model for the task. Models in MII are sourced from multiple open-source projects
            such as Huggingface Transformer, FairSeq, EluetherAI etc. For the list of supported models for each task, please
            see here [TODO].

        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

        deployment_type: One of the ``enum mii.DeploymentTypes: [LOCAL]``.
            *``LOCAL`` uses a grpc server to create a local deployment, and query the model must be done by creating a query handle using
              `mii.mii_query_handle` and posting queries using ``mii_request_handle.query`` API,

        model_path: Optional: In LOCAL deployments this is the local path where model checkpoints are available. In AML deployments this
            is an optional relative path with AZURE_MODEL_DIR for the deployment.

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        enable_zero: Optional: Defaults to False. Use this flag to enable or disable DeepSpeed-ZeRO inference

        ds_config: Optional: Defaults to None. Use this to specify the DeepSpeed configuration when enabling DeepSpeed-ZeRO inference

        force_register_model: Optional: Defaults to False. For AML deployments, set it to True if you want to re-register your model
            with the same ``aml_model_tags`` using checkpoints from ``model_path``.

        mii_config: Optional: Dictionary specifying optimization and deployment configurations that should override defaults in ``mii.config.MIIConfig``.
            mii_config is future looking to support extensions in optimization strategies supported by DeepSpeed Inference as we extend mii.
            As of now, it can be used to set tensor-slicing degree using 'tensor_parallel' and port number for deployment using 'port_number'.

        version: Optional: Version to be set for AML deployment, useful if you want to deploy the same model with different settings.
    Returns:
        If deployment_type is `LOCAL`, returns just the name of the deployment that can be used to create a query handle using `mii.mii_query_handle(deployment_name)`

    """
    # TODO: parse dictionary and create MIIConfig
    if mii_config is None:
        config = {}
    if isinstance(mii_config, str):
        with open(mii_config, "r") as f:
            config_dict = json.load(f)
    elif isinstance(mii_config, dict):
        config_dict = config
    else:
        raise ValueError(f"'mii_config' argument expected str or dict, got type {type(mii_config)}")

    # Update with values from kwargs
    overlap_keys = set(config_dict.keys()).intersection(kwargs.keys())
    for key in overlap_keys:
        # If keys dont match, raise value error
        if config_dict[key] != kwargs[key]:
            raise ValueError(f"Conflicting argument '{key}' in 'config':{config_dict[key]} and kwargs:{kwargs[key]}")
    # Update config dict with kwargs values set by user
    config_dict.update(kwargs)
    # parse and validate mii config
    mii_config = mii.config.MIIConfig(**mii_config)

    task = mii.utils.get_task(mii_config.task)

    if not mii_config.skip_model_check:
        mii.utils.check_if_task_and_model_is_valid(task, model)
        if mii_config.enable_deepspeed:
            mii.utils.check_if_task_and_model_is_supported(task, model)

    if mii_config.enable_deepspeed:
        logger.info(
            "************* MII is using DeepSpeed Optimizations to accelerate your model *************"
        )
    else:
        logger.info(
            "************* DeepSpeed Optimizations not enabled. Please use enable_deepspeed to get better performance *************"
        )

    # add fields for replica deployment
    lb_config = None
    if mii_config.enable_load_balancing:
        replica_pool = _allocate_processes(mii_config.hostfile,
                                           mii_config.tensor_parallel,
                                           mii_config.replica_num)
        replica_configs = []
        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            port_offset = 1 if mii_config.enable_load_balancing else 0
            base_port = mii_config.port_number + i * mii_config.tensor_parallel + port_offset
            tensor_parallel_ports = list(
                range(base_port,
                      base_port + mii_config.tensor_parallel))
            torch_dist_port = mii_config.torch_dist_port + i
            replica_configs.append(
                ReplicaConfig(hostname=hostname,
                              tensor_parallel_ports=tensor_parallel_ports,
                              torch_dist_port=torch_dist_port,
                              gpu_indices=gpu_indices))
        lb_config = LoadBalancerConfig(port=mii_config.port_number,
                                       replica_configs=replica_configs)

    create_score_file(deployment_name=deployment_name,
                      deployment_type=mii_config.deployment_type,
                      task=task,
                      model_name=model,
                      ds_optimize=mii_config.enable_deepspeed,
                      ds_zero=mii_config.enable_zero,
                      ds_config=mii_config.ds_config,
                      mii_config=mii_config,
                      model_path=mii_config.model_path,
                      lb_config=lb_config)

    if mii_config.deployment_type == DeploymentType.AML:
        _deploy_aml(deployment_name=mii_config.deployment_name, model_name=model, version=mii_config.version)
    elif mii_config.deployment_type == DeploymentType.LOCAL:
        return _deploy_local(mii_config.deployment_name, model_path=model_path)
    else:
        raise Exception(f"Unknown deployment type: {deployment_type}")


def _deploy_local(deployment_name, model_path):
    mii.utils.import_score_file(deployment_name).init()


def _deploy_aml(deployment_name, model_name, version):
    acr_name = mii.aml_related.utils.get_acr_name()
    mii.aml_related.utils.generate_aml_scripts(acr_name=acr_name,
                                               deployment_name=deployment_name,
                                               model_name=model_name,
                                               version=version)
    print(
        f"AML deployment assets at {mii.aml_related.utils.aml_output_path(deployment_name)}"
    )
    print("Please run 'deploy.sh' to bring your deployment online")


def _allocate_processes(hostfile_path, tensor_parallel, num_replicas):
    resource_pool = fetch_hostfile(hostfile_path)
    assert resource_pool is not None and len(
        resource_pool) > 0, f'No hosts found in {hostfile_path}'

    replica_pool = []
    allocated_num = 0
    for host, slots in resource_pool.items():
        available_on_host = slots
        while available_on_host >= tensor_parallel:
            if allocated_num >= num_replicas:
                break
            if slots < tensor_parallel:
                raise ValueError(
                    f'Host {host} has {slots} slot(s), but {tensor_parallel} slot(s) are required'
                )

            allocated_num_on_host = slots - available_on_host
            replica_pool.append(
                (host,
                 [
                     i for i in range(allocated_num_on_host,
                                      allocated_num_on_host + tensor_parallel)
                 ]))
            allocated_num += 1

            available_on_host -= tensor_parallel

    if allocated_num < num_replicas:
        raise ValueError(
            f'No sufficient GPUs for {num_replicas} replica(s), only {allocated_num} replica(s) can be deployed'
        )

    return replica_pool
