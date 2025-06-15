import torch.multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from tokasaurus.common_types import ServerConfig, TimedBarrier
from tokasaurus.utils import error_propogation_decorator, setup_logging


@dataclass
class DownloadRequest:
    cartridge_id: str
    source: str
    force_redownload: bool
    cartridges_path: Path


@dataclass
class DownloadComplete:
    cartridge_id: str
    success: bool
    force_redownload: bool
    error_message: str | None = None


@error_propogation_decorator
def start_download_worker(
    config: ServerConfig,
    q_download_requests: mp.Queue,
    q_download_complete: mp.Queue,
    process_name: str,
    barrier: TimedBarrier,
):
    setup_logging(config)
    
    from loguru import logger
    bound_logger = logger.bind(process_name=process_name)
    bound_logger.info("Download worker started")
    
    barrier.wait()
    
    while True:
        request: DownloadRequest = q_download_requests.get()
        
        try:
            # Import here to avoid issues when wandb is not available
            from tokasaurus.manager.cartridge_downloader import download_cartridge
            
            download_cartridge(
                cartridge_id=request.cartridge_id,
                source=request.source,
                cartridges_path=request.cartridges_path,
                force_redownload=request.force_redownload,
                logger=bound_logger
            )
            
            response = DownloadComplete(
                cartridge_id=request.cartridge_id,
                success=True,
                force_redownload=request.force_redownload
            )
            bound_logger.info(f"Downloaded cartridge {request.cartridge_id}")
            
        except Exception as e:
            response = DownloadComplete(
                cartridge_id=request.cartridge_id,
                success=False,
                force_redownload=request.force_redownload,
                error_message=str(e)
            )
            bound_logger.error(f"Failed to download {request.cartridge_id}: {e}")
        
        q_download_complete.put(response) 