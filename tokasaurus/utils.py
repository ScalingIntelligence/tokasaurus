import cProfile
import multiprocessing.connection as mp_conn
import os
import pickle
import queue
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING
import re

import psutil
import requests
import torch
import torch.distributed
import torch.multiprocessing as mp
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from transformers import GenerationConfig

if TYPE_CHECKING:
    from tokasaurus.common_types import ServerConfig


@contextmanager
def timer(
    name: str, enable: bool = True, min_ms: float | None = None, profile: bool = False
):
    if not enable:
        yield
        return

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    start = time.time()

    yield

    end = time.time()

    if profile:
        profiler.disable()

    ms = (end - start) * 1000
    if min_ms is None or ms > min_ms:
        print(f"timer {name}: {ms:.2f}ms")
        if profile:
            profiler.print_stats()


@contextmanager
def profile(name: str):
    """
    use built in python profiler
    """
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()
    print(f"Profiling {name}:")
    profiler.print_stats()


def get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def set_rank(rank: int):
    os.environ["LOCAL_RANK"] = str(rank)


def set_world_size(world_size: int):
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)


def set_master_port(port: int):
    os.environ["MASTER_PORT"] = str(port)


def get_master_port() -> int:
    return int(os.environ.get("MASTER_PORT", "29500"))


def set_master_addr(addr: str):
    os.environ["MASTER_ADDR"] = addr


def get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR", "localhost")


def is_local() -> bool:
    return get_rank() == 0


def lprint(*args, **kwargs):
    if is_local():
        print(*args, **kwargs)


def ltqdm(*args, **kwargs):
    if is_local():
        return tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs, disable=True)


def lprint_tensor(tensor: torch.Tensor):
    lprint(tensor.sum(), tensor.view(-1)[:5], tensor.view(-1)[-5:])


def std(lst):
    mean = sum(lst) / len(lst)
    return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5


def median(lst):
    sorted_lst = sorted(lst)
    if len(sorted_lst) % 2 == 1:
        return sorted_lst[len(sorted_lst) // 2]
    else:
        return (
            sorted_lst[len(sorted_lst) // 2 - 1] + sorted_lst[len(sorted_lst) // 2]
        ) / 2


@dataclass
class TimeResult:
    times: list[float]
    warmup_times: list[float]
    cpu_times: list[float]
    cpu_warmup_times: list[float]

    def mean(self):
        return sum(self.times) / len(self.times)

    def std(self):
        return std(self.times)

    def cpu_mean(self):
        return sum(self.cpu_times) / len(self.cpu_times)

    def cpu_std(self):
        return std(self.cpu_times)

    def median(self):
        return median(self.times)

    def cpu_median(self):
        return median(self.cpu_times)

    def fancy_table(self):
        cpu_mean = self.cpu_mean()
        cpu_std = self.cpu_std()
        cpu_median = self.cpu_median()
        cpu_min = min(self.cpu_times)
        cpu_max = max(self.cpu_times)

        gpu_mean = self.mean()
        gpu_std = self.std()
        gpu_median = self.median()
        gpu_min = min(self.times)
        gpu_max = max(self.times)

        # table with cpu and gpu rows, and columns for mean, std, median, min, max
        table = [
            ["CPU", cpu_mean, cpu_std, cpu_median, cpu_min, cpu_max],
            ["GPU", gpu_mean, gpu_std, gpu_median, gpu_min, gpu_max],
        ]
        return tabulate(
            table,
            headers=["Metric", "Mean", "Std", "Median", "Min", "Max"],
            tablefmt="github",
        )


def convert_unit(milis: float, unit: str) -> float:
    if unit == "ms":
        return milis
    elif unit == "s":
        return milis / 1000
    elif unit == "us":
        return milis * 1000
    else:
        raise ValueError(f"Invalid unit: {unit}")


@torch.no_grad()
def timed(
    fn,
    num_iters=50,
    num_warmup=10,
    unit: str = "ms",
    between_fn=None,
    prog=True,
    barrier=False,
):
    warmup_times = []
    times = []
    cpu_warmup_times = []
    cpu_times = []
    for itr in tqdm(range(num_iters + num_warmup), desc="Timing", disable=not prog):
        if between_fn is not None:
            between_fn()
            torch.cuda.synchronize()

        if barrier:
            torch.distributed.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  # type: ignore
        cpu_start = time.perf_counter()
        _ = fn()
        cpu_end = time.perf_counter()
        end.record()  # type: ignore
        torch.cuda.synchronize()

        if barrier:
            torch.distributed.barrier()

        gpu_milis = start.elapsed_time(end)
        cpu_milis = (cpu_end - cpu_start) * 1000

        gpu_time = convert_unit(gpu_milis, unit)
        cpu_time = convert_unit(cpu_milis, unit)

        if itr >= num_warmup:
            times.append(gpu_time)
            cpu_times.append(cpu_time)
        else:
            warmup_times.append(gpu_time)
            cpu_warmup_times.append(cpu_time)

    return TimeResult(
        times=times,
        warmup_times=warmup_times,
        cpu_times=cpu_times,
        cpu_warmup_times=cpu_warmup_times,
    )


@torch.no_grad()
def timed_with_graph(
    fn,
    num_iters=50,
    num_warmup=10,
    unit: str = "ms",
    prog=True,
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # type: ignore
        # warmup
        for _ in ltqdm(range(3)):
            fn()

    torch.cuda.current_stream().wait_stream(s)

    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    warmup_times = []
    times = []
    cpu_warmup_times = []
    cpu_times = []
    for itr in tqdm(range(num_iters + num_warmup), desc="Timing", disable=not prog):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_start = time.perf_counter()
        g.replay()
        cpu_end = time.perf_counter()
        end.record()
        torch.cuda.synchronize()

        gpu_milis = start.elapsed_time(end)
        cpu_milis = (cpu_end - cpu_start) * 1000

        gpu_time = convert_unit(gpu_milis, unit)
        cpu_time = convert_unit(cpu_milis, unit)

        if itr >= num_warmup:
            times.append(gpu_time)
            cpu_times.append(cpu_time)
        else:
            warmup_times.append(gpu_time)
            cpu_warmup_times.append(cpu_time)

    return TimeResult(
        times=times,
        warmup_times=warmup_times,
        cpu_times=cpu_times,
        cpu_warmup_times=cpu_warmup_times,
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            child.terminate()

        gone, alive = psutil.wait_procs(children, timeout=5)

        for p in alive:
            p.kill()

        parent.terminate()
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass


@dataclass
class ServerArgs:
    args: list[str]
    start_server: bool = True
    port: int | None = None


@contextmanager
def sglang_manager(config: ServerArgs):
    if not config.start_server:
        yield None
        return config.port

    if config.args is None:
        args = ""
    else:
        str_args = [str(arg) for arg in config.args]
        args = " ".join(str_args)

    port = find_free_port()

    launch_command = f"""python -m sglang.launch_server \
        --port {port} \
        {args}"""

    print(f"Starting sglang server with command: {launch_command}")
    server_process = subprocess.Popen(launch_command, shell=True)
    print(f"Started sglang server with pid {server_process.pid}")

    try:
        wait_for_ping(port, server_process, max_retries=500, ping_endpoint="health")
        yield port
    finally:
        print(f"Killing sglang server (pid {server_process.pid})...")
        terminate_process_tree(server_process.pid)
        print("Done killing sglang server.")


@contextmanager
def vllm_manager(config: ServerArgs):
    if not config.start_server:
        yield None
        return config.port

    if config.args is None:
        args = ""
    else:
        str_args = [str(arg) for arg in config.args]
        args = " ".join(str_args)

    port = find_free_port()

    vllm_command = f"""python vllm_server.py \
        --port {port} \
        {args}"""

    print(f"Starting vllm server with command: {vllm_command}")
    vllm_process = subprocess.Popen(vllm_command, shell=True)
    print(f"Started vllm server with pid {vllm_process.pid}")

    try:
        wait_for_ping(port, vllm_process, max_retries=500)
        yield port
    finally:
        print(f"Killing vllm server (pid {vllm_process.pid})...")
        terminate_process_tree(vllm_process.pid)
        print("Done killing vllm server.")


def wait_for_ping(
    port,
    popen: subprocess.Popen,
    retry_seconds=2,
    max_retries=500,
    ping_endpoint: str = "ping",
):
    # wait for the server to start, by /ping-ing it
    print(f"Waiting for server to start on port {port}...")
    for i in range(max_retries):
        try:
            requests.get(f"http://localhost:{port}/{ping_endpoint}")
            return
        except requests.exceptions.ConnectionError:
            if popen.poll() is not None:
                raise RuntimeError(
                    f"Server died with code {popen.returncode} before starting."
                )

            print(f"Server not yet started (attempt {i}) retrying...")
            time.sleep(retry_seconds)

    raise RuntimeError(f"Server not started after {max_retries} attempts.")


def gpus_to_cvd(gpus: list[int]):
    return "CUDA_VISIBLE_DEVICES=" + ",".join([str(x) for x in gpus])


def save_pkl(obj, path: str):
    """Save an object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: str):
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_eos_token_ids(generation_config: GenerationConfig):
    model_eos = generation_config.eos_token_id
    if model_eos is None:
        eos_token_ids = []
    elif isinstance(model_eos, int):
        eos_token_ids = [model_eos]
    else:
        assert isinstance(model_eos, list)
        eos_token_ids = model_eos

    return set(eos_token_ids)


def setup_logging(config: "ServerConfig"):
    logger.remove()

    def log_filter(record):
        if (log_procs := config.log_procs) is not None:
            return record["extra"]["process_name"] in log_procs
        return True

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "{extra[process_name]} | "
        "<level>{message}</level>",
        filter=log_filter,
        level=config.log_level,
    )


def queue_iterator(q: mp.Queue):
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:
            break


def block_on_queues(queues: list[mp.Queue]):
    readers = [q._reader for q in queues]
    mp_conn.wait(readers, timeout=None)


def error_propogation_decorator(func):
    """
    Sometimes, for weird reasons, error messages are
    silenced in engine subprocesses. Here we explicitly
    print to stdout to ensure the error is visible.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("Caught error - reprinting and reraising")
            print("-" * 80)
            print(e)
            traceback.print_exc()
            print("-" * 80)
            raise

    return wrapper


def sanitize_cartridge_id(cartridge_id: str) -> str:
    """
    Sanitize a cartridge ID to make it safe for use as a directory name.
    
    This function:
    - Replaces forward slashes and backslashes with dashes
    - Replaces other problematic characters (colon, pipe, etc.) with underscores
    - Removes/replaces path traversal sequences
    - Ensures the result is a valid directory name
    
    Args:
        cartridge_id: The original cartridge ID
        
    Returns:
        A sanitized cartridge ID safe for use in directory paths
        
    Raises:
        ValueError: If the cartridge_id is empty or results in an empty string after sanitization
    """
    if not cartridge_id or not cartridge_id.strip():
        raise ValueError("Cartridge ID cannot be empty")
    
    # Replace forward slashes and backslashes with dashes
    sanitized = cartridge_id.replace("/", "-").replace("\\", "-")
    
    # Replace other problematic characters with underscores
    # This includes: colon, pipe, question mark, asterisk, angle brackets, quotes
    sanitized = re.sub(r'[:|*?<>"\'\x00-\x1f\x7f]', "_", sanitized)
    
    # Handle path traversal sequences more aggressively
    # Replace any sequence that starts with dots
    sanitized = re.sub(r'\.+', "_", sanitized)
    
    # Remove any remaining slashes that might have been missed
    sanitized = sanitized.replace("/", "-").replace("\\", "-")
    
    # Trim whitespace and replace multiple consecutive special chars with single underscore
    sanitized = re.sub(r'[_-]{2,}', "_", sanitized.strip())
    
    # Ensure it's not empty after sanitization
    if not sanitized or sanitized in ["_"]:
        sanitized = f"sanitized_{abs(hash(cartridge_id)) % 1000000}"
    
    return sanitized
