import logging
from pathlib import Path

def set_logging(log_dir : Path):
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_name = Path("run.log")
    log_path = log_dir / log_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()]
    )