import threading
from tkinter import Menu, Tk, Scale
import tkinter as tk
from typing import Final, Optional

from predict.predict import Predictor
import simpleaudio as sa

from ui.menu_toolbar import MenuToolbar


class GUI:
    FONT_LARGE = ("Roman", 18)
    FONT_SMALL = ("Roman", 16)

    def __init__(self) -> None:
        self._predictor = Predictor()
        self._main_window: Tk = self._create_main_window()
        self._user_input: tk.Entry = self._create_user_input_section()
        self._n_verses_scale: tk.Scale = Scale(
            self._main_window,
            from_=2,
            to=12,  # TODO - add scrolling
            length=200,
            width=25,
            orient="horizontal",
            label="Number of verses",
            font=self.FONT_LARGE,
        )
        self._n_verses_scale.set(6)
        self._n_verses_scale.grid(row=2, column=0, padx=10, pady=10)
        submit_button = tk.Button(
            self._main_window,
            text="Submit",
            command=self._on_submit_action,
            width=10,
            height=1,
            font=self.FONT_SMALL,
        )
        submit_button.grid(row=2, column=2, padx=10, pady=10)
        self._sentiment_label = tk.Label(
            self._main_window,
            text="You seem to be experiencing...",
            anchor="w",
            justify="left",
            font=self.FONT_LARGE,
        )
        self._sentiment_label.grid(row=4, column=0, padx=10, pady=10, columnspan=3)

        self._is_playing: bool = False
        self._song: Optional[sa.PlayObject] = None
        self._play_song_button = tk.Button(
            self._main_window,
            text="Play song",
            command=self._change_playing_state,
            state="disabled",
            width=10,
            height=1,
            font=self.FONT_SMALL,
        )
        self._play_song_button.grid(row=5, column=0, padx=10, pady=10, columnspan=3)
        self._lyrics_label: tk.Label = tk.Label(
            self._main_window,
            text="",
            anchor="w",
            justify="left",
            font=self.FONT_LARGE,
        )
        self._lyrics_label.grid(row=6, column=0, padx=10, pady=10, columnspan=3)

    def _create_main_window(self) -> Tk:
        WINDOW_TITLE: Final[str] = "Moodsic"
        MIN_WINDOW_WIDTH_IN_PIXELS: Final[int] = 600
        MIN_WINDOW_HEIGHT_IN_PIXELS: Final[int] = 720

        main_window: Tk = Tk()
        main_window.title(WINDOW_TITLE)
        main_window.minsize(MIN_WINDOW_WIDTH_IN_PIXELS, MIN_WINDOW_HEIGHT_IN_PIXELS)
        menu_bar: Menu = MenuToolbar(main_window).create()
        main_window.config(menu=menu_bar)

        return main_window

    def _refresh_view(self) -> None:
        self._lyrics_label.config(text="\n".join(self._predictor.lyrics))
        # TODO - treat NEUTRAL case
        self._sentiment_label.config(
            text=f"You seem to be experiencing {self._predictor.sentiment.upper()}"
        )

    def _on_submit_action(self) -> None:
        prompt = self._user_input.get("1.0", "end-1c")
        n_verses = int(self._n_verses_scale.get())
        self._predictor.run(prompt, n_verses)
        self._refresh_view()
        self._play_song_button["state"] = "normal"

    def _change_playing_state(self) -> None:
        self._is_playing = not self._is_playing
        if self._is_playing:
            wave_obj = sa.WaveObject.from_wave_file("../Outputs/name.wav")
            self._song = wave_obj.play()
            self._play_song_button.config(text="Stop")
        else:
            self._song.stop()
            self._play_song_button.config(text="Play")

    def _create_user_input_section(self) -> None:
        label = tk.Label(self._main_window, text="How are you feeling today ?", font=self.FONT_LARGE)
        label.grid(row=0, column=0, padx=10, pady=10, columnspan=3)
        user_input = tk.Text(self._main_window, height=2, width=60, font=self.FONT_SMALL)
        user_input.grid(row=1, column=0, padx=10, pady=10, columnspan=3)
        return user_input

    def run(self) -> None:
        thread = threading.Thread(target=self._predictor.load_artifacts)
        thread.start()
        self._main_window.mainloop()
