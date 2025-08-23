from contextlib import contextmanager

import pydra
import torch.multiprocessing as mp

from tokasaurus.common_types import Engine, ProcessInfo, ServerConfig, TimedBarrier
from tokasaurus.manager.manager import start_manager
from tokasaurus.model.entry import get_model_process_dict
from tokasaurus.server.endpoints import start_server
from tokasaurus.utils import find_free_port


def cleanup_processes(processes: list[mp.Process]):
    for process in processes:
        process.kill()

    for process in processes:
        process.join()


def make_engine(config: ServerConfig, dp_rank: int, master_port: int):
    q_server_to_manager = mp.Queue()
    q_manager_to_server = mp.Queue()

    q_manager_to_model = mp.Queue()
    q_model_to_manager = mp.Queue()
    
    # Download worker queues
    q_download_requests = mp.Queue()
    q_download_complete = mp.Queue()

    # Start the model process
    process_dict = get_model_process_dict(
        config=config,
        q_manager_to_model=q_manager_to_model,
        q_model_to_manager=q_model_to_manager,
        dp_rank=dp_rank,
        master_port=master_port,
    )
    
    # Add download worker process
    from tokasaurus.manager.download_worker import start_download_worker
    process_dict["download_worker"] = ProcessInfo(
        target=start_download_worker,
        kwargs={
            "config": config,
            "q_download_requests": q_download_requests,
            "q_download_complete": q_download_complete,
        },
    )
    
    process_dict["manager"] = ProcessInfo(
        target=start_manager,
        kwargs={
            "config": config,
            "q_manager_to_model": q_manager_to_model,
            "q_model_to_manager": q_model_to_manager,
            "q_server_to_manager": q_server_to_manager,
            "q_manager_to_server": q_manager_to_server,
            "q_download_requests": q_download_requests,
            "q_download_complete": q_download_complete,
            "dp_rank": dp_rank,
        },
    )

    if config.dp_size > 1:
        process_dict = {f"dp{dp_rank}_{k}": v for k, v in process_dict.items()}

    engine = Engine(
        q_server_to_manager=q_server_to_manager,
        q_manager_to_server=q_manager_to_server,
        proc_dict=process_dict,
    )

    return engine


def make_proc_dict(config: ServerConfig, add_extra_barrier_member: bool = False):
    master_port = find_free_port()
    engines = [
        make_engine(config, dp_rank, master_port) for dp_rank in range(config.dp_size)
    ]

    pooled_proc_dict: dict[str, ProcessInfo] = {}
    for engine in engines:
        for proc_name, proc_info in engine.proc_dict.items():
            assert proc_name not in pooled_proc_dict
            pooled_proc_dict[proc_name] = proc_info

    pooled_proc_dict["server"] = ProcessInfo(
        target=start_server,
        kwargs={
            "config": config,
            "engines": engines,
        },
    )

    num_procs = len(pooled_proc_dict)
    barrier_size = num_procs + 1 if add_extra_barrier_member else num_procs
    barrier = TimedBarrier(barrier_size, "System startup time")

    for proc_name, proc_info in pooled_proc_dict.items():
        proc_info.kwargs["barrier"] = barrier
        proc_info.kwargs["process_name"] = proc_name

    return pooled_proc_dict, barrier


@contextmanager
def server_manager(config: ServerConfig, finalize=True):
    mp.set_start_method("spawn", force=True)

    if finalize:
        config.finalize()

    process_dict, barrier = make_proc_dict(config, add_extra_barrier_member=True)

    processes = []

    try:
        for _, process_info in process_dict.items():
            p = process_info.make_process()
            p.start()
            processes.append(p)

        barrier.wait()

        yield
    finally:
        cleanup_processes(processes)


def start(config: ServerConfig):
    mp.set_start_method("spawn", force=True)

    process_dict, _ = make_proc_dict(config, add_extra_barrier_member=False)

    print(f"Starting {len(process_dict)} processes: {list(process_dict.keys())}")
    print(f"Running in the main process: {config.local_proc_name}")

    processes = []
    for proc_name, process_info in process_dict.items():
        if proc_name == config.local_proc_name:
            continue
        p = process_info.make_process()
        p.start()
        processes.append(p)

    try:
        local_proc = process_dict[config.local_proc_name]
        local_proc.target(*local_proc.args, **local_proc.kwargs)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating all subprocesses.")
        cleanup_processes(processes)
        print("... All subprocesses cleaned up.")


def main():
    """
    For use as a setup.py entry point.
    """
    pydra.run(start)


if __name__ == "__main__":
    main()
