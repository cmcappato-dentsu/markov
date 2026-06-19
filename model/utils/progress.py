from tqdm import tqdm

def get_progress_bar(total: int, desc: str):
    try:
        pbar = tqdm(total=total, desc=desc, unit="ch", dynamic_ncols=True)
        return pbar, True
    except Exception:
        return None, False