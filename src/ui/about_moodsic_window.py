from tkinter import Tk, Toplevel, Label, PhotoImage, TOP, BOTTOM
from typing import Final, Literal


class AboutMoodsicWindow:
    NORMAL_FONT_SIZE: Final[int] = 14
    X_PADDING: Final[int] = 20
    Y_PADDING: Final[int] = 20
    WINDOW_SIZE_IN_PIXELS: Final[int] = 380  # square window
    CENTER_ANCHOR: Final[Literal["center"]] = "center"

    def __init__(self, mainWindow: Tk):
        self.__aboutMoodsicWindow: Toplevel = self.__createAboutMoodsicWindow(mainWindow)
        self.__displayLogo()
        self.__displayContent()

    def __createAboutMoodsicWindow(self, mainWindow: Tk) -> Toplevel:
        ABOUT_MOODSIC_WINDOW_TITLE: Final[str] = "About Moodsic"

        aboutMoodsicWindow: Toplevel = Toplevel()
        aboutMoodsicWindow.title(ABOUT_MOODSIC_WINDOW_TITLE)
        aboutMoodsicWindow.minsize(self.WINDOW_SIZE_IN_PIXELS, self.WINDOW_SIZE_IN_PIXELS)
        aboutMoodsicWindow.maxsize(self.WINDOW_SIZE_IN_PIXELS, self.WINDOW_SIZE_IN_PIXELS)
        aboutMoodsicWindow.transient(mainWindow)
        aboutMoodsicWindow.resizable(False, False)
        return aboutMoodsicWindow

    def __displayLogo(self) -> None:
        HALF_SIZE: Final[int] = 2

        logo: PhotoImage = PhotoImage(file="ui/logo.png").subsample(HALF_SIZE, HALF_SIZE)
        logoLabel: Label = Label(self.__aboutMoodsicWindow, image=logo, anchor=self.CENTER_ANCHOR)
        logoLabel.image = logo
        logoLabel.pack(side=TOP)

    def __displayContent(self) -> None:
        content: Final[
            str
        ] = """Listen to yourself. Listen to your feelings
Est. 2024
HAIDE U!"""
        Label(
            self.__aboutMoodsicWindow,
            text=content,
            font=("Roman", self.NORMAL_FONT_SIZE),
            width=self.WINDOW_SIZE_IN_PIXELS,
            anchor=self.CENTER_ANCHOR,
        ).pack(side=BOTTOM, padx=self.X_PADDING, pady=self.Y_PADDING)
