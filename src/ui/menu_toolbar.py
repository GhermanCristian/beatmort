from tkinter import Menu, Tk
from typing import Final

from ui.about_moodsic_window import AboutMoodsicWindow
from ui.help_window import HelpWindow
from ui.settings_window import SettingsWindow


class MenuToolbar:
    def __init__(self, main_window: Tk) -> None:
        self._main_window = main_window

    def _settings_command(self) -> None:
        SettingsWindow(self._main_window)

    def _help_command(self) -> None:
        HelpWindow(self._main_window)

    def _about_command(self) -> None:
        AboutMoodsicWindow(self._main_window)

    def create(self) -> Menu:
        SETTINGS_COMMAND_LABEL: Final[str] = "Settings"
        HELP_COMMAND_LABEL: Final[str] = "Help"
        ABOUT_MOODSIC_COMMAND_LABEL: Final[str] = "About Moodsic"

        menu_bar: Menu = Menu(self._main_window)
        menu_bar.add_command(label=SETTINGS_COMMAND_LABEL, command=self._settings_command)
        menu_bar.add_command(label=HELP_COMMAND_LABEL, command=self._help_command)
        menu_bar.add_command(label=ABOUT_MOODSIC_COMMAND_LABEL, command=self._about_command)
        return menu_bar
