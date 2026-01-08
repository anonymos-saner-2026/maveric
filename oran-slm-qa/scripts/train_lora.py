import argparse
from pathlib import Path
import yaml
from oran_qa.training.sft import run_sft_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_sft_lora(cfg)


if __name__ == "__main__":
    main()
