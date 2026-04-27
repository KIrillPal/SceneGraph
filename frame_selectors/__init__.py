from .base import (
    BaseFrameSelector,
    BaseSelector,
    FrameEmbeddings,
    FrameMasks,
    FrameSample,
)


def __getattr__(name: str):
    if name == "MaskCoverageSelector":
        from .mask_coverage import MaskCoverageSelector

        return MaskCoverageSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseFrameSelector",
    "BaseSelector",
    "FrameEmbeddings",
    "FrameMasks",
    "FrameSample",
    "MaskCoverageSelector",
]
