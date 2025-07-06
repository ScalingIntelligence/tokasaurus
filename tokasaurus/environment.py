import platform


def is_flashinfer_supported() -> bool:
    """
    Returns True if flashinfer (and Triton) are likely usable on this platform.
    Currently, Triton does not support macOS.
    """
    return platform.system() != "Darwin"


def assert_flashinfer_supported():
    """
    Raises a RuntimeError with a clear message if the current platform is not supported by flashinfer.
    """
    if not is_flashinfer_supported():
        raise RuntimeError(
            "Tokasaurus currently depends on flashinfer, which in turn requires 'triton'.\n"
            "However, 'triton' is not supported on macOS.\n"
            "Please run Tokasaurus in a Linux environment with a supported GPU."
        )
