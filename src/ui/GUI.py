import threading
from tkinter import Tk, Scale
import tkinter as tk
from typing import Final

from predict.predict import Predictor
import simpleaudio as sa


class GUI:
    FONT = ("Roman", 18)
    FONT_INPUT = ("Roman", 16)

    def __init__(self) -> None:
        self._predictor = Predictor()
        self._main_window: Tk = self.__create_main_window()
        self._user_input: tk.Entry = self._create_user_input_section()
        submit_button = tk.Button(self._main_window, text="Submit", command=self._on_submit_action)
        submit_button.pack()
        self._n_verses_scale: tk.Scale = Scale(
            self._main_window, from_=2, to=32, length=200, width=25
        )
        self._n_verses_scale.pack()
        self._lyrics_label: tk.Label = tk.Label(
            self._main_window,
            text="Lyrics will end up here",
            anchor="w",
            justify="left",
            font=self.FONT,
        )
        self._lyrics_label.pack()

        self._is_playing: bool = False
        self._play_song_button = tk.Button(
            self._main_window, text="Play", command=self._change_playing_state
        )
        self._play_song_button.pack()

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
        n_verses = int(self._n_verses_scale.get())
        self._predictor.run(prompt, n_verses)
        self._refresh_view()

    def _change_playing_state(self) -> None:
        self._is_playing = not self._is_playing
        if self._is_playing:
            wave_obj = sa.WaveObject.from_wave_file("../Outputs/test.wav")
            wave_obj.play()
        else:
            pass

    def _create_user_input_section(self) -> None:
        label = tk.Label(self._main_window, text="How are you feeling today ?", font=self.FONT)
        label.pack()
        user_input = tk.Entry(self._main_window, width=100, font=self.FONT_INPUT)
        user_input.pack()
        return user_input

    def run(self) -> None:
        thread = threading.Thread(target=self._predictor.load_artifacts)
        thread.start()
        self._main_window.mainloop()
