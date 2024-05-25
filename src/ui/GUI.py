import threading
from tkinter import Tk
import tkinter as tk
from typing import Final

from predict.predict import Predictor


class GUI:
    def __init__(self) -> None:
        self._predictor = Predictor()
        self._main_window: Tk = self.__create_main_window()
        self._user_input: tk.Entry = self._create_user_input_section()
        submit_button = tk.Button(
            self._main_window, text="Submit", command=lambda: self._on_submit_action()
        )
        submit_button.pack()
        self._lyrics_label: tk.Label = tk.Label(self._main_window, text="Lyrics will end up here")
        self._lyrics_label.pack()

    def __create_main_window(self) -> Tk:
        WINDOW_TITLE: Final[str] = "apptitle"
        MIN_WINDOW_WIDTH_IN_PIXELS: Final[int] = 1100
        MIN_WINDOW_HEIGHT_IN_PIXELS: Final[int] = 480

        main_window: Tk = Tk()
        main_window.title(WINDOW_TITLE)
        main_window.minsize(MIN_WINDOW_WIDTH_IN_PIXELS, MIN_WINDOW_HEIGHT_IN_PIXELS)

        return main_window

    def _refresh_view(self) -> None:
        self._lyrics_label.config(text="\n".join(self._predictor.lyrics))

    def _on_submit_action(self) -> None:
        prompt = self._user_input.get()
        self._predictor.run(prompt)
        self._refresh_view()

    def _create_user_input_section(self) -> None:
        label = tk.Label(self._main_window, text="What's on your mind ?")
        label.pack()
        user_input = tk.Entry(self._main_window)
        user_input.pack()
        return user_input

    def run(self) -> None:
        thread = threading.Thread(target=self._predictor.load_artifacts)
        thread.start()
        self._main_window.mainloop()
