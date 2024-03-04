import logging
import multiprocessing
import os
import subprocess
from itertools import islice
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd
from hydra.core.utils import setup_globals
from rich.logging import RichHandler
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

setup_globals()

try:
    import tables

    HDF = True
except:
    HDF = False


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def read_log(
    paths: str,
    loading_functions: List[Callable],
    processing_functions: List[Callable],
    outpath: Path,
    batch_size: int = 10,
    n_processes: int = 4,
) -> pd.DataFrame:
    log = logging.getLogger("ReadLogs")
    filenames = []
    if type(paths) == str:
        paths = [paths]
    paths = list(set(paths))  # filter duplicates
    for path in paths:
        path = Path(path)
        filenames.extend(list(path.glob(f"**/*")))

    hdf_key = "run_data"
    batch_names = []
    for i, batch in enumerate(batched(filenames, batch_size)):
        log.info(f"Batch {i}: Start reading {len(batch)} logs from {paths}")
        df = map_multiprocessing(
            task_functions=loading_functions,
            task_params=batch,
            task_string="Reading logs...",
            n_processes=n_processes,
        )
        log.info("Concatenating logs...")
        df = pd.concat(df).reset_index(drop=True)
        log.info("Postprocess logs...")
        for f in processing_functions:
            processed = f(df)
            if processed is not None:
                df = processed
        data_fn_tmp = str(outpath) + f"_{i}"
        batch_names.append(data_fn_tmp)
        log.info(f"Dumping logs to '{data_fn_tmp}'.")
        if HDF:
            df.to_hdf(data_fn_tmp, hdf_key)
        else:
            df.to_csv(f"{data_fn_tmp}.csv")
        log.info(f"Done with batch {i} 🙂")

    log.info("Collect all batches and save to disk")
    if HDF:
        df = pd.concat([pd.read_hdf(fn, hdf_key) for fn in batch_names])
        df.to_hdf(outpath, hdf_key)
        for fn in batch_names:
            subprocess.Popen(f"rm {fn}")
    else:
        df = pd.concat([pd.read_csv(f"{fn}.csv") for fn in batch_names])
        df.to_csv(f"{outpath}.csv")
        for fn in batch_names:
            os.remove(f"{fn}.csv")
    log.info(f"Done 🙂")
    return df


def map_multiprocessing(
    task_functions: List[Callable],
    task_params: list[Any],
    n_processes: int = 4,
    task_string: str = "Working...",
) -> list:
    results = []

    def get_results(path):
        res = []
        for f in task_functions:
            results = map(f, path)
            results = list(filter(lambda x: x is not None, results))
            if len(results) > 0:
                res.append(pd.concat(results).reset_index(drop=True))
        return res

    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        refresh_per_second=2,
    ) as progress:
        task_id = progress.add_task(f"[cyan]{task_string}", total=len(task_params))
        try:
            with multiprocessing.Pool(processes=n_processes) as pool:
                for result in pool.imap(get_results, task_params):
                    results.append(result)
                    progress.advance(task_id)
        except AttributeError:
            print("Pickling failed, falling back on sequential loading.")
            for param in task_params:
                results.extend(get_results([param]))
                progress.advance(task_id)
    return results


def read_logs(
    data_path: str,
    loading_functions: List[Callable],
    processing_functions: List[Callable],
    save_to: str,
) -> pd.DataFrame:
    path = Path(data_path)
    filenames = list(path.glob(f"**/*"))
    outpath = Path(save_to)
    return read_log(filenames, loading_functions, processing_functions, outpath)


def get_missing_jobs(
    data_path: str, is_done: Callable, n_processes: int = 4
) -> pd.DataFrame:
    path = Path(data_path)
    filenames = list(path.glob(f"**/*"))
    results = []

    def is_missing(filename):
        if is_done(filename) is False and is_done(filename) is not None:
            return filename
        else:
            return None

    results = list(filter(lambda x: x is not None, map(is_missing, filenames)))
    log = logging.getLogger("CheckJobs")
    log.info(f"Found {len(results)} missing jobs.")
    log.info(f"That means {len(filenames) - len(results)} jobs are done.")
    return results
