import os
import sys
import yaml
import logging

from datetime import datetime
import pandas as pd

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner

from eagle.log import setup_simple_log
from eagle.utils import open_yaml_config

logger = logging.getLogger("eagle")

def date_to_str(
    date: str,
) -> str:
    """
    Format initializtion date to a nicely formatted string that is used for writing out files.

    Args:
        date (str): date of initialization.

    Returns:
        str -- date of initialization in str format.
    """
    dt = datetime.fromisoformat(date)
    return dt.strftime("%Y-%m-%dT%H")

def create_config(
    init_date: str,
    main_config: dict,
) -> dict:
    """
    Create the config that will be passed to anemoi-inference.
    As of right now the "extract_lam" functionality is really only used to save out a static lam file when wanted.
    This could be easily updated if we think we would ever be interested in running inference over just CONUS.

    Args:
        init_date (str): date of initialization.
        extract_lam (bool): logic to extract lam domain, or run whole nested domain.
        lead_time (int): desired lead time to save out. Default=LEAD_TIME.
        lam (bool): true/false indication if LAM. Default=LAM.
        checkpoint_path (str): path to checkpoint. Default=CHECKPOINT.
        input_data_path (str): path to input data when not using LAM (e.g. you trained on 1 source). Default=INPUT_DATA_PATH.
        lam_path (str): path to regional data when using LAM. Default=LAM_PATH.
        global_path (str: path to global data when using LAM. Default=GLOBAL_PATH.
        output_path (str): path for saving files. Default=OUTPUT_PATH.

    Returns:
        dict -- config for anemoi-inference.
    """
    date_str = date_to_str(date=init_date)

    lead_time = main_config.get("lead_time")
    config = {
        "checkpoint": main_config["checkpoint_path"],
        "date": date_str,
        "lead_time": lead_time,
        "input": {"dataset": main_config["input_dataset_kwargs"]},
    }
    if main_config.get("extract_lam", False):
        config["output"] = {
            "extract_lam": {
                "output": {
                    "netcdf": {
                        "path": f"{main_config['output_path']}/lam.nc",
                    },
                },
            },
        }
    else:
        config["output"] = {
            "netcdf": f"{main_config['output_path']}/{date_str}.{lead_time}h.nc",
        }

    return config


def run_forecast(
    init_date: str,
    main_config: dict,
) -> None:
    """
    Inference pipeline.

    Args:
        init_date (str): date of initialization.

    Returns:
        None -- files saved out to output path.
    """
    config_dict = create_config(
        init_date=init_date,
        main_config=main_config,
    )
    config = RunConfiguration.load(config_dict)
    runner = create_runner(config)
    runner.execute()

    return


def run_inference(config_filename=None):
    if len(sys.argv) != 2 and config_filename is None:
        raise Exception("Did not get an argument. Usage is:\npython run_inference.py recipe.yaml")
        sys.exit(1)

    config_filename = sys.argv[1] if config_filename is None else config_filename
    setup_simple_log()
    main_config = open_yaml_config(config_filename)

    dates = pd.date_range(start=main_config["start_date"], end=main_config["end_date"], freq=main_config["freq"])

    logger.info(f"Starting inference with initial condition dates:\n{dates}")

    nc_files = []
    for d in dates:
        run_forecast(
            init_date=str(d),
            main_config=main_config,
        )
        logger.info(f"Done with {d}")
