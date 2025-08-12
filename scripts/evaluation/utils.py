import time
import json
import logging
import jsonlines
from tqdm import tqdm
from typing import Union
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute(func, input_list_or_num_samples: Union[list, int], output_path: str, max_workers: int, logger: logging.Logger):
    out = open(output_path, 'a')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if isinstance(input_list_or_num_samples, list):
            futures = [executor.submit(func, item) for item in input_list_or_num_samples]
        else:
            futures = [executor.submit(func) for _ in range(input_list_or_num_samples)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                res: dict = future.result()
            except Exception as e:
                logger.info(f"[error] {e}")
                continue
            if res:
                out.write(json.dumps(res, ensure_ascii=False) + '\n')
                out.flush()
    out.close()

def retry(max: int=10, sleep: int=1, logger: logging.Logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"[retry] try {i} times")
                    if i == max - 1:
                        raise Exception("Error: {}. Retry {} failed after {} times".format(e, func.__name__, max))
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

def init_logger(log_path: str, log_name: str):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger