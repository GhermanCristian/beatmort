from pathlib import Path
from tkinter import Button, Label, Tk, Toplevel, filedialog
from typing import Final


class SettingsWindow:
    X_PADDING: Final[int] = 15
    Y_PADDING: Final[int] = 15
    NORMAL_FONT_SIZE: Final[int] = 12
    DOWNLOAD_LOCATION_ROW_INDEX: Final[int] = 1

    def __init__(self, main_window: Tk) -> None:
        self._main_window = main_window
        self._settings_window = self._create_settings_window()
        self._create_download_location_info()
        self._download_location_field: Label = self._create_download_location_field()
        self._create_choose_new_location_button()

    def _create_settings_window(self) -> Toplevel:
        SETTINGS_WINDOW_TITLE: Final[str] = "Settings"
        SETTINGS_WINDOW_MIN_WIDTH_IN_PIXELS: Final[int] = 640
        SETTINGS_WINDOW_MIN_HEIGHT_IN_PIXELS: Final[int] = 480

        settings_window: Toplevel = Toplevel()
        settings_window.title(SETTINGS_WINDOW_TITLE)
        settings_window.minsize(
            SETTINGS_WINDOW_MIN_WIDTH_IN_PIXELS, SETTINGS_WINDOW_MIN_HEIGHT_IN_PIXELS
        )
        settings_window.transient(self._main_window)
        return settings_window

    def _create_download_location_info(self) -> None:
        DOWNLOAD_LOCATION_INFO_TEXT: Final[str] = "Download location"
        DOWNLOAD_LOCATION_INFO_COLUMN_SPAN: Final[int] = 3

        download_location_info_label: Label = Label(
            self._settings_window,
            text=DOWNLOAD_LOCATION_INFO_TEXT,
            font=("Roman", self.NORMAL_FONT_SIZE),
        )
        download_location_info_label.grid(
            row=self.DOWNLOAD_LOCATION_ROW_INDEX,
            columnspan=DOWNLOAD_LOCATION_INFO_COLUMN_SPAN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )

    def _create_download_location_field(self) -> Label:
        DOWNLOAD_LOCATION_FIELD_COLUMN: Final[int] = 4
        DOWNLOAD_LOCATION_FIELD_COLUMN_SPAN: Final[int] = 10

        downloadLocationField: Label = Label(
            self._settings_window,
            text=str(Path(".").resolve()),
            font=("Roman", self.NORMAL_FONT_SIZE),
        )  # bg=utilsGUI.CREAM_COLOR)
        downloadLocationField.grid(
            row=self.DOWNLOAD_LOCATION_ROW_INDEX,
            column=DOWNLOAD_LOCATION_FIELD_COLUMN,
            columnspan=DOWNLOAD_LOCATION_FIELD_COLUMN_SPAN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )
        return downloadLocationField

    def _create_choose_new_location_button(self) -> None:
        CHOOSE_NEW_LOCATION_BUTTON_TEXT: Final[str] = "Select"
        CHOOSE_NEW_LOCATION_BUTTON_COLUMN: Final[int] = 14

        choose_new_location_button: Button = Button(
            self._settings_window,
            text=CHOOSE_NEW_LOCATION_BUTTON_TEXT,
            command=self._set_new_download_location,
            font=("Roman", self.NORMAL_FONT_SIZE),
        )
        choose_new_location_button.grid(
            row=self.DOWNLOAD_LOCATION_ROW_INDEX,
            column=CHOOSE_NEW_LOCATION_BUTTON_COLUMN,
            padx=self.X_PADDING,
            pady=self.Y_PADDING,
        )

    def _select_base_download_location(self) -> str:
        DOWNLOAD_LOCATION_DIALOG_TITLE: Final[str] = "Select the download location"
        return filedialog.askdirectory(
            initialdir=str(Path(".").resolve()), title=DOWNLOAD_LOCATION_DIALOG_TITLE
        )

    def _set_new_download_location(self) -> None:
        new_download_location: str = self._select_base_download_location()
        self._download_location_field.config(text=new_download_location)
        settingsProcessor.setDownloadLocation(new_download_location)
