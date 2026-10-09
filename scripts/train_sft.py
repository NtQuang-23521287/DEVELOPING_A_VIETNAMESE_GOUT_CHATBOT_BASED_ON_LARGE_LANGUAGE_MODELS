import argparse
from gout_llmops.core.config import load_yaml
from gout_llmops.training.sft import train_sft

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/sft_v1.yaml")
    args = ap.parse_args()
    train_sft(load_yaml(args.config))
