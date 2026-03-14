import os
from dataclasses import replace

from dotenv import load_dotenv

from src.config import ExperimentConfig

load_dotenv()
from src.main import main
from src.analysis import compare_weight_configs


def get_openai_key():
    return os.getenv("OPENAI_API_KEY", None)


if __name__ == "__main__":
    base_cfg = ExperimentConfig(
        openai_api_key=get_openai_key(),
        use_api=True,
        use_optimization=False,
        max_queries=10,
    )

    datasets = ["nsir"]

    if base_cfg.analysis_mode:
        for ds in datasets:
            main(replace(base_cfg, dataset=ds))
    else:
        weight_experiments = [
            # {"reg_weight": 0.0, "pos_weight": 0.0, "neg_weight": 0.0,
            #  "optimization_steps": 0, "use_optimization": False},
            {"reg_weight": 0.2, "pos_weight": 1.0, "neg_weight": 1.0,
             "optimization_steps": 20, "use_optimization": True},
        ]

        for dataset in datasets:
            for weights in weight_experiments:
                cfg = replace(base_cfg, dataset=dataset, **weights)
                main(cfg)

            compare_weight_configs(dataset, base_cfg.checkpoint_dir)
