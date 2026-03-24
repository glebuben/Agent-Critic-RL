from src.sac import train_sac
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent with specified config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml")

    args = parser.parse_args()

    train_sac(args.config)