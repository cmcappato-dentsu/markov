import logging

def setup_logger(verbosity: str = "info"):
    level = logging.INFO
    
    if str(verbosity).lower() == "debug":
        level = logging.DEBUG
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )