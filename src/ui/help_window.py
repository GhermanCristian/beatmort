from tkinter import Toplevel, Tk, Label
from typing import Final, List


class HelpWindow:
    FONT_SIZE: Final[int] = 14
    CENTER_ANCHOR: Final[str] = "center"
    X_PADDING: Final[int] = 20
    Y_PADDING: Final[int] = 20

    def __init__(self, main_window: Tk):
        self._help_window: Toplevel = self._create_help_window(main_window)
        self._display_text()

    def _create_help_window(self, main_window: Tk) -> Toplevel:
        HELP_WINDOW_TITLE: Final[str] = "Help"
        HELP_WINDOW_MIN_WIDTH_IN_PIXELS: Final[int] = 720
        HELP_WINDOW_MIN_HEIGHT_IN_PIXELS: Final[int] = 360

        help_window: Toplevel = Toplevel()
        help_window.title(HELP_WINDOW_TITLE)
        help_window.minsize(HELP_WINDOW_MIN_WIDTH_IN_PIXELS, HELP_WINDOW_MIN_HEIGHT_IN_PIXELS)
        help_window.transient(main_window)
        return help_window

    def _display_text(self) -> None:
        content: Final[List[str]] = [
            "Tell your virtual therapist how you feel and select how long you want the lyrics to be,",
            "then press submit. After a short while, the lyrics will be displayed on screen and saved",
            "to disk at the given location. The 'play' button will become active, so that you can",
            "listen to your feels in real time!"
            "\n",
            "Prompt tips - use mostly adjectives. After all, that's how you describe things ;)",
            "\n",
            "How does the sentiment influence the output music ? Simple. Positive feelings lead to",
            "faster, more upbeat songs or instruments, whereas negative feelings may feel",
            "gloomy and slow"
        ]
        Label(self._help_window, text="\n".join(content), font=("Roman", self.FONT_SIZE)).pack(
            padx=self.X_PADDING, pady=self.Y_PADDING
        )
