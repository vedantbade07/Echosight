"""
Accessibility-first overlay UI: high-contrast caption bar, adjustable
size/position, and status indicators. This layer is original to
EchoSight — it is not part of any Qualcomm AI Hub sample.
"""


class CaptionOverlay:
    def __init__(self, font_size: int = 28, high_contrast: bool = True):
        self.font_size = font_size
        self.high_contrast = high_contrast

    def show_caption(self, text: str) -> None:
        # TODO: render via a lightweight always-on-top window
        # (e.g. PyQt/PySide or a native WinUI overlay).
        raise NotImplementedError

    def show_status(self, message: str) -> None:
        raise NotImplementedError
