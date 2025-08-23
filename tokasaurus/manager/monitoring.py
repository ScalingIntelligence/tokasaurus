import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache, wraps
from statistics import mean, stdev
from typing import TYPE_CHECKING

import statsd
from tabulate import tabulate
import wandb

from tokasaurus.common_types import ServerConfig

if TYPE_CHECKING:
    from tokasaurus.manager.types import (
        ManagerState,
        ScheduleDecision,
    )

TRACK_TIME = int(os.environ.get("TRACK_TIME", "0")) > 0
SIMPLE_TRACK_TIME = int(os.environ.get("SIMPLE_TRACK_TIME", "0")) > 0


TIME_TRACKER = defaultdict(lambda: 0.0)
COUNT_TRACKER = defaultdict(lambda: 0)


@contextmanager
def track_time(key: str):
    if not TRACK_TIME:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        TIME_TRACKER[key] += time.time() - start
        COUNT_TRACKER[key] += 1


def track_time_decorator(key: str | None = None):
    def decorator(func):
        if not TRACK_TIME:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            with track_time(key or func.__name__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def simple_timer(name: str, enable: bool = True):
    if not SIMPLE_TRACK_TIME or not enable:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        ms = (time.time() - start) * 1000
        print(f"{name} took {ms:.2f}ms")


def simple_decorator(name: str, enable: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with simple_timer(name, enable):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@cache
def _statsd_client(server_url: str):
    """Create a statsd client."""
    return statsd.StatsClient(server_url)


@cache
def _hostname() -> str:
    """Get the hostname of this machine, used to identify it in statsd logs."""
    return socket.gethostname().replace(".", "_")

class WandbLogger:
    def __init__(self, config: ServerConfig, dp_rank: int = 0):
        self.config = config
        self.dp_rank = dp_rank
        
        if not config.wandb_enabled:
            self.run = None
            return
            
        # Initialize WandB with configuration, including DP rank in tags
        init_kwargs = {
            "entity": config.wandb_entity,
            "project": config.wandb_project,
            "config": config.to_dict(),
            "tags": []
        }
        
        
        # Add DP rank to run name for differentiation
        # Build a comprehensive run name
        name_parts = []
        
        # Add model name (extract just the model name from path)
        model_name = config.model.split('/')[-1]
        name_parts.append(model_name)
        
        # Add hostname
        hostname = _hostname()
        name_parts.append(hostname)
        
        # Add DP rank if using multiple data parallel workers
        if config.dp_size > 1:
            name_parts.append(f"dp{dp_rank}")
        
        # Add user-specified run name if provided
        if config.wandb_run_name:
            name_parts.append(config.wandb_run_name)
        
        # Add GPU type
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')
                name_parts.append(gpu_name)
        except:
            pass  # Skip GPU type if we can't determine it
        
        init_kwargs["name"] = "_".join(name_parts).lower()
            
        self.run = wandb.init(**init_kwargs)
    
    def log(self, metrics: dict[str, int | float]):
        if self.run is not None:
            self.run.log(metrics)
    
    def close(self):
        if self.run is not None:
            self.run.finish()


def log_to_statsd(config: ServerConfig, metrics: dict[str, int | float]):
    """Log a dictionary of metrics to statsd."""
    if config.statsd_server_url is None:
        return
        
    prefix = f"tokasaurus.{_hostname()}.port_{config.port}"
    client = _statsd_client(config.statsd_server_url)

    for k, v in metrics.items():
        prefix_with_k = f"{prefix}.{k}"
        client.gauge(prefix_with_k, v)


def log_to_wandb(wandb_logger: WandbLogger | None, metrics: dict[str, int | float]):
    """Log a dictionary of metrics to wandb."""
    if wandb_logger is not None:
        wandb_logger.log(metrics)


@dataclass
class HydragenStats:
    num_grouped_blocks: int
    num_total_blocks: int


def maybe_stdev(values):
    if len(values) < 2:
        return -1

    return stdev(values)


@dataclass
class StatsTracker:
    decisions: list["ScheduleDecision"] = field(default_factory=list)
    manager_idle_times: list[float] = field(default_factory=list)
    num_new_commands_list: list[int] = field(default_factory=list)
    num_steps_to_schedule_list: list[int] = field(default_factory=list)
    hydragen_stats: list[HydragenStats] = field(default_factory=list)
    num_finished_seqs: int = 0
    num_finished_reqs: int = 0
    time_since_reset: float = -1
    global_num_decisions: int = 0
    global_num_prefill_tokens: int = 0
    global_num_decode_tokens: int = 0
    global_num_finished_seqs: int = 0
    global_num_finished_reqs: int = 0

    def elapsed_time(self):
        assert self.is_initialized()
        return time.time() - self.time_since_reset

    def reset(self):
        self.time_since_reset = time.time()
        self.decisions.clear()
        self.manager_idle_times.clear()
        self.num_new_commands_list.clear()
        self.num_steps_to_schedule_list.clear()
        self.hydragen_stats.clear()
        self.num_finished_seqs = 0
        self.num_finished_reqs = 0

    def add_decision(self, decision: "ScheduleDecision"):
        assert self.is_initialized()
        self.decisions.append(decision)
        self.global_num_decisions += 1
        self.global_num_prefill_tokens += decision.num_prefill_tokens()
        self.global_num_decode_tokens += decision.num_decoding_tokens()

    def add_finished_seq(self):
        assert self.is_initialized()
        self.num_finished_seqs += 1
        self.global_num_finished_seqs += 1

    def add_finished_req(self):
        assert self.is_initialized()
        self.num_finished_reqs += 1
        self.global_num_finished_reqs += 1

    def add_manager_idle_time(self, wait_time: float):
        self.manager_idle_times.append(wait_time)

    def add_num_new_commands(self, num_new_commands: int):
        self.num_new_commands_list.append(num_new_commands)

    def add_num_steps_to_schedule(self, num_steps_to_schedule: int):
        self.num_steps_to_schedule_list.append(num_steps_to_schedule)

    def add_hydragen_stats(self, num_grouped_blocks: int, num_total_blocks: int):
        self.hydragen_stats.append(HydragenStats(num_grouped_blocks, num_total_blocks))

    def calc_stats(self):
        assert self.is_initialized()

        elapsed = self.elapsed_time()

        total_decode_tokens = sum(
            decision.num_decoding_tokens() for decision in self.decisions
        )
        total_prefill_tokens = sum(
            decision.num_prefill_tokens() for decision in self.decisions
        )
        total_tokens = total_decode_tokens + total_prefill_tokens

        prefill_tps = total_prefill_tokens / elapsed
        decode_tps = total_decode_tokens / elapsed
        total_tps = total_tokens / elapsed

        num_decisions = len(self.decisions)

        manager_total_idle_time = sum(self.manager_idle_times)
        assert manager_total_idle_time <= elapsed
        manager_idle_frac = manager_total_idle_time / elapsed

        seqs_per_second = self.num_finished_seqs / elapsed
        reqs_per_second = self.num_finished_reqs / elapsed

        stats = {
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_tps": total_tps,
            "seqs_per_second": seqs_per_second,
            "reqs_per_second": reqs_per_second,
            "prefill_per_forward": total_prefill_tokens / num_decisions,
            "decode_per_forward": total_decode_tokens / num_decisions,
            "toks_per_forward": total_tokens / num_decisions,
            "forwards_per_second": num_decisions / elapsed,
            "manager_idle_frac": manager_idle_frac,
            "manager_mean_num_commands": mean(self.num_new_commands_list),
            "manager_mean_num_steps_to_schedule": mean(self.num_steps_to_schedule_list),
            "manager_max_num_steps_to_schedule": max(self.num_steps_to_schedule_list),
            "manager_std_num_commands": maybe_stdev(self.num_new_commands_list),
            "manager_std_num_steps_to_schedule": maybe_stdev(
                self.num_steps_to_schedule_list
            ),
            "num_decisions": num_decisions,
            "num_finished_seqs": self.num_finished_seqs,
            "num_finished_reqs": self.num_finished_reqs,
            "global_num_decisions": self.global_num_decisions,
            "global_num_prefill_tokens": self.global_num_prefill_tokens,
            "global_num_decode_tokens": self.global_num_decode_tokens,
            "global_num_finished_seqs": self.global_num_finished_seqs,
            "global_num_finished_reqs": self.global_num_finished_reqs,
            "elapsed_time": elapsed,
        }

        if len(self.hydragen_stats) > 0:
            grouped_blocks = sum(
                stat.num_grouped_blocks for stat in self.hydragen_stats
            )
            total_blocks = sum(stat.num_total_blocks for stat in self.hydragen_stats)

            if total_blocks > 0:
                frac = grouped_blocks / total_blocks
            else:
                frac = 0.0

            stats["hydragen_sharing_frac"] = frac

        else:
            stats["hydragen_sharing_frac"] = 0.0

        return stats

    def is_initialized(self):
        return self.time_since_reset != -1


def format_value(value: int):
    str_value = str(value)
    num_digits = len(str_value)

    digit_map = {
        3: "",
        6: "K",
        9: "M",
        12: "B",
        15: "T",
        18: "Q",
    }

    for pow10, suffix in digit_map.items():
        if num_digits <= pow10:
            if pow10 == 3:
                val = value
            else:
                val = round(value / 10 ** (pow10 - 3), 2)
            return f"{val}{suffix}"

    return f"{str_value} 👀"


@track_time_decorator()
def step_stats(
    state: "ManagerState",
    manager_idle_time: float,
    num_new_commands: int,
    num_steps_to_schedule: int,
):
    config = state.config
    stats_tracker = state.stats_tracker

    stats_tracker.add_manager_idle_time(manager_idle_time)
    stats_tracker.add_num_new_commands(num_new_commands)
    stats_tracker.add_num_steps_to_schedule(num_steps_to_schedule)

    elapsed_time = stats_tracker.elapsed_time()

    if elapsed_time >= config.stats_report_seconds and len(stats_tracker.decisions) > 0:
        tracker_stats = stats_tracker.calc_stats()

        kv_allocated = state.block_allocator.fraction_used()

        kv_reserved = (
            sum(
                seq.expected_num_additional_blocks(config.page_size, add_buffer=True)
                for seq in state.scheduling_queue.running_seqs()
            )
            / config.kv_cache_num_blocks()
        )

        kv_total_used = kv_allocated + kv_reserved

        running_seq_lens = [
            seq.total_scheduled() for seq in state.scheduling_queue.running_seqs()
        ]
        mean_running_seq_len = mean(running_seq_lens) if running_seq_lens else 0

        # combination of accumulated stats from the tracker (e.g. throughput)
        # and measurements of the current state (e.g. number of running
        # sequences right now)
        stats = {
            **tracker_stats,
            "kv_cache_frac_allocated": kv_allocated,
            "kv_cache_frac_reserved": kv_reserved,
            "kv_cache_frac_total": kv_total_used,
            "num_running_seqs": state.scheduling_queue.num_running_seqs(),
            "num_decoding_seqs": len(state.scheduling_queue.decoding_seqs),
            "num_prefilling_seqs": len(state.scheduling_queue.prefilling_seqs),
            "num_queued_seqs": len(state.scheduling_queue.queued_seqs),
            "mean_running_seq_len": mean_running_seq_len,
        }

        logger_parts = [
            f"Throughput: {stats['total_tps']:.2f} tok/s ({stats['prefill_tps']:.2f} prefill, {stats['decode_tps']:.2f} decode), {stats['toks_per_forward']:.2f} tok/fwd ({stats['decode_per_forward']:.2f} decode, {stats['prefill_per_forward']:.2f} prefill), {stats['forwards_per_second']:.2f} fwd/s, {stats['seqs_per_second']:.2f} seqs/s, {stats['reqs_per_second']:.2f} reqs/s",
            f"Seqs (#): {stats['num_running_seqs']} running ({stats['num_decoding_seqs']} decoding, {stats['num_prefilling_seqs']} prefilling, avg seq len {stats['mean_running_seq_len']:.2f}), {stats['num_queued_seqs']} queued",
            f"KV Cache Usage: {stats['kv_cache_frac_allocated'] * 100:.2f}% allocated, {stats['kv_cache_frac_reserved'] * 100:.2f}% reserved, ({stats['kv_cache_frac_total'] * 100:.2f}% total)",
            f"Per Manager Loop: Idle {stats['manager_idle_frac'] * 100:.2f}%, {stats['manager_mean_num_commands']:.2f} ± {stats['manager_std_num_commands']:.2f} commands, {stats['manager_mean_num_steps_to_schedule']:.2f} ± {stats['manager_std_num_steps_to_schedule']:.2f} steps scheduled (max {stats['manager_max_num_steps_to_schedule']})",
        ]

        if (tracker := state.early_stopping_tracker) is not None:
            if not tracker.buffer_empty():
                stats["early_stopping_frac_mean"] = tracker.buffer_mean()
                stats["early_stopping_frac_std"] = tracker.buffer_std()
                stats["early_stopping_buffer_len"] = tracker.buffer_len()
            else:
                stats["early_stopping_frac_mean"] = 0.0
                stats["early_stopping_frac_std"] = 0.0
                stats["early_stopping_buffer_len"] = 0

            logger_parts.append(
                f"Early Stopping: {stats['early_stopping_frac_mean']:.2f} ± {stats['early_stopping_frac_std']:.2f} (buffer size {stats['early_stopping_buffer_len']}, warmup_done={tracker.is_warmed_up()})"
            )

        if state.config.use_hydragen:
            logger_parts.append(
                f"Hydragen: {stats['hydragen_sharing_frac'] * 100:.2f}% decode block sharing"
            )

        logger_parts.extend(
            [
                f"Reporting Interval: {stats['elapsed_time']:.2f}s, {stats['num_decisions']} forward passes, {stats['num_finished_seqs']} finished seqs, {stats['num_finished_reqs']} finished reqs",
                f"Lifetime Counters: {format_value(stats['global_num_decisions'])} forward passes, {format_value(stats['global_num_prefill_tokens'])} prefill tokens, {format_value(stats['global_num_decode_tokens'])} decode tokens, {format_value(stats['global_num_finished_seqs'])} finished seqs, {format_value(stats['global_num_finished_reqs'])} finished reqs",
            ]
        )

        log_to_statsd(config, stats)
        log_to_wandb(getattr(state, 'wandb_logger', None), stats)

        state.logger.info(" | ".join(logger_parts))

        if TRACK_TIME:
            time_table = []
            for key, cum_time in TIME_TRACKER.items():
                call_count = COUNT_TRACKER[key]
                time_table.append([key, cum_time, call_count, cum_time / call_count])
            time_table.append(["total", sum(TIME_TRACKER.values())])
            time_table.append(["idle", sum(state.stats_tracker.manager_idle_times)])
            time_table.append(["elapsed", elapsed_time])
            time_table.sort(key=lambda x: x[1])
            state.logger.info(
                "Execution time by tracker key:\n"
                + tabulate(
                    time_table,
                    headers=["Key", "Time (s)", "# Calls", "Avg Per Call (s)"],
                )
            )

            TIME_TRACKER.clear()
            COUNT_TRACKER.clear()

        state.stats_tracker.reset()
