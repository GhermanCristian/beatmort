from pathlib import Path
from tkinter import Button, Label, Tk, Toplevel, filedialog
from typing import Final

from constants import Constants


class SettingsWindow:
    X_PADDING: Final[int] = 15
    Y_PADDING: Final[int] = 15
    NORMAL_FONT_SIZE: Final[int] = 12
    SAVE_LOCATION_ROW_INDEX: Final[int] = 1

    def __init__(self, main_window: Tk) -> None:
        self._main_window = main_window
        self._settings_window = self._create_settings_window()
        self._create_save_location_info()
        self._save_location_field: Label = self._create_save_location_field()
        self._create_choose_new_location_button()

    def _create_settings_window(self) -> Toplevel:
        SETTINGS_WINDOW_TITLE: Final[str] = "Settings"
        SETTINGS_WINDOW_MIN_WIDTH_IN_PIXELS: Final[int] = 600
        SETTINGS_WINDOW_MIN_HEIGHT_IN_PIXELS: Final[int] = 200

        settings_window: Toplevel = Toplevel()
        settings_window.title(SETTINGS_WINDOW_TITLE)
        settings_window.minsize(
            SETTINGS_WINDOW_MIN_WIDTH_IN_PIXELS, SETTINGS_WINDOW_MIN_HEIGHT_IN_PIXELS
        )
        settings_window.transient(self._main_window)
        return settings_window

    def _create_save_location_info(self) -> None:
        SAVE_LOCATION_INFO_TEXT: Final[str] = "Save location"
        SAVE_LOCATION_INFO_COLUMN_SPAN: Final[int] = 3

        save_location_info_label: Label = Label(
            self._settings_window,
            text=SAVE_LOCATION_INFO_TEXT,
            font=("Roman", self.NORMAL_FONT_SIZE),
        )
        save_location_info_label.grid(
            row=self.SAVE_LOCATION_ROW_INDEX,
            columnspan=SAVE_LOCATION_INFO_COLUMN_SPAN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )

    def _create_save_location_field(self) -> Label:
        SAVE_LOCATION_FIELD_COLUMN: Final[int] = 4
        SAVE_LOCATION_FIELD_COLUMN_SPAN: Final[int] = 10

        saveLocationField: Label = Label(
            self._settings_window,
            text=str(Path(Constants.OUTPUT_SAVE_DIR).resolve()),
            font=("Roman", self.NORMAL_FONT_SIZE),
            bg=("#f5e6c2"),
        )
        saveLocationField.grid(
            row=self.SAVE_LOCATION_ROW_INDEX,
            column=SAVE_LOCATION_FIELD_COLUMN,
            columnspan=SAVE_LOCATION_FIELD_COLUMN_SPAN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )
        return saveLocationField

    def _create_choose_new_location_button(self) -> None:
        CHOOSE_NEW_LOCATION_BUTTON_TEXT: Final[str] = "Select"
        CHOOSE_NEW_LOCATION_BUTTON_COLUMN: Final[int] = 14

        choose_new_location_button: Button = Button(
            self._settings_window,
            text=CHOOSE_NEW_LOCATION_BUTTON_TEXT,
            command=self._set_new_save_location,
            font=("Roman", self.NORMAL_FONT_SIZE),
        )
        choose_new_location_button.grid(
            row=self.SAVE_LOCATION_ROW_INDEX,
            column=CHOOSE_NEW_LOCATION_BUTTON_COLUMN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )

    def _select_base_save_location(self) -> str:
        SAVE_LOCATION_DIALOG_TITLE: Final[str] = "Select the save location"
        return filedialog.askdirectory(
            initialdir=str(Path(Constants.OUTPUT_SAVE_DIR).resolve()),
            title=SAVE_LOCATION_DIALOG_TITLE,
        )

    def _set_new_save_location(self) -> None:
        new_save_location: str = self._select_base_save_location()
        self._save_location_field.config(text=new_save_location)
        Constants.OUTPUT_SAVE_DIR = new_save_location
