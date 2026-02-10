import logging
import os
from datetime import datetime

def setup_logger(name):
    # current date and time
    now = datetime.now()
    date_folder = now.strftime("%d_%m")
    timestamp = now.strftime("%H_%M_%S")

    # create date-wise log directory
    log_dir= os.path.join("logs",date_folder)
    os.makedirs(log_dir,exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid Duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    #log file
    log_file = os.path.join(log_dir,f"{name}_{timestamp}.log")

    #file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)


    #Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    #formatter 
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    #Attach Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger