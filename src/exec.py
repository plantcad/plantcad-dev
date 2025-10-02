import logging
import time
import os
import ray

from thalas.execution import Executor, ExecutorMainConfig, ExecutorStep, InputName
from thalas.utilities.ray_utils import is_local_ray_cluster

logger = logging.getLogger("ray")


# TODO: Move to Thalas once the initial Marin extraction is complete;
# This was copied from https://github.com/marin-community/marin/blob/fe373c233ee7288cbf8e7600765c3fc6fb6fa3ac/src/marin/execution/executor.py#L1094
# and modified only to remove the draccus.wrap decorator and compulsory logging config
def executor_main(
    config: ExecutorMainConfig,
    steps: list[ExecutorStep | InputName],
    description: str | None = None,
    init_logging: bool = True,
) -> Executor:
    """Main entry point for experiments (to standardize)"""
    if init_logging:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    time_in = time.time()
    ray.init(
        namespace="thalas",
        ignore_reinit_error=True,
        resources={"head_node": 1} if is_local_ray_cluster() else None,
    )  # We need to init ray here to make sure we have the correct namespace for actors
    # (status_actor in particular)
    time_out = time.time()
    logger.info(f"Ray init took {time_out - time_in:.2f}s")
    time_in = time.time()

    prefix = config.prefix
    if prefix is None:
        # infer from the environment
        if "THALAS_PREFIX" in os.environ:
            prefix = os.environ["THALAS_PREFIX"]
        else:
            raise ValueError(
                "Must specify a prefix or set the THALAS_PREFIX environment variable"
            )
    elif "THALAS_PREFIX" in os.environ:
        if prefix != os.environ["THALAS_PREFIX"]:
            logger.warning(
                f"THALAS_PREFIX environment variable ({os.environ['THALAS_PREFIX']}) is different from the "
                f"specified prefix ({prefix})"
            )

    executor_info_base_path = config.executor_info_base_path
    if executor_info_base_path is None:
        # infer from prefix
        executor_info_base_path = os.path.join(prefix, "experiments")

    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
        description=description,
    )

    executor.run(
        steps=steps,
        dry_run=config.dry_run,
        run_only=config.run_only,
        force_run_failed=config.force_run_failed,
    )
    time_out = time.time()
    logger.info(f"Executor run took {time_out - time_in:.2f}s")
    # print json path again so it's easy to copy
    logger.info(f"Executor info written to {executor.executor_info_path}")
    logger.info(f"View the experiment at {executor.get_experiment_url()}")

    return executor
