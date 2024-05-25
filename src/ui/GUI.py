from tkinter import Tk
import tkinter as tk
from typing import Final

from predict.predict import Predictor


class GUI:
    def __init__(self) -> None:
        self._predictor = Predictor()
        self._predictor.load_artifacts()  # TODO - do in separate thread

    def __create_main_window(self) -> Tk:
        WINDOW_TITLE: Final[str] = "apptitle"
        MIN_WINDOW_WIDTH_IN_PIXELS: Final[int] = 1100
        MIN_WINDOW_HEIGHT_IN_PIXELS: Final[int] = 480

        main_window: Tk = Tk()
        main_window.title(WINDOW_TITLE)
        main_window.minsize(MIN_WINDOW_WIDTH_IN_PIXELS, MIN_WINDOW_HEIGHT_IN_PIXELS)

        return main_window

    def _on_submit_action(self, user_input: tk.Entry) -> None:
        prompt = user_input.get()
        lyrics = self._predictor.run(prompt)
        for l in lyrics:
            print(l)

    def _create_user_input_section(self, main_window: Tk) -> None:
        tk.Label(main_window, text="What's on your mind ?").grid(row=0)
        user_input = tk.Entry(main_window)
        user_input.grid(row=0, column=1)
        tk.Button(main_window, text="Submit", command=lambda: self._on_submit_action(user_input)).grid(
            row=3, column=1, sticky=tk.W, pady=4
        )

    def run(self) -> None:
        main_window: Tk = self.__create_main_window()
        self._create_user_input_section(main_window)
        main_window.mainloop()
