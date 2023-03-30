from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from math import ceil, log10
from game import Game
from pg import ParityGame
from mpg import MeanPayoffGame
from dpg import DiscountedPayoffGame
from ssg import SimpleStochasticGame
from eg import EnergyGame
from os import remove
from os.path import join, dirname, realpath, isfile
from datetime import datetime
import numpy as np

zoom_factor = 2 ** (1 / 4)


class Gui:
    def __init__(self):

        self.top = Tk()

        self.top.title("")

        self.menubar = Menu(self.top)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.gen_menu = Menu(self.filemenu, tearoff=0)
        self.gen_menu.add_command(
            label="PG",
            command=lambda: self.gen_prompt(ParityGame),
            accelerator="Ctrl-1",
        )
        self.gen_menu.add_command(
            label="MPG",
            command=lambda: self.gen_prompt(MeanPayoffGame),
            accelerator="Ctrl-2",
        )
        self.gen_menu.add_command(
            label="EG",
            command=lambda: self.gen_prompt(EnergyGame),
            accelerator="Ctrl-3",
        )
        self.gen_menu.add_command(
            label="DPG",
            command=lambda: self.gen_prompt(DiscountedPayoffGame),
            accelerator="Ctrl-4",
        )
        self.gen_menu.add_command(
            label="SSG",
            command=lambda: self.gen_prompt(SimpleStochasticGame),
            accelerator="Ctrl-5",
        )
        self.filemenu.add_cascade(label="Generate", menu=self.gen_menu)
        self.filemenu.add_command(
            label="Open", accelerator="Ctrl-O", command=self.open_file
        )
        self.filemenu.add_command(
            label="Save", accelerator="Ctrl-S", command=self.save_file_as
        )
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.top.config(menu=self.menubar)

        self.frame_left = Frame(self.top)

        self.frame_top_left = Frame(self.frame_left)
        self.frame_bottom_left = Frame(self.frame_left)

        self.scale = 1.0
        self.glob_opt = Button(
            master=self.frame_top_left,
            command=lambda: self._render(opt=True),
            text="Show optimal",
            state="disabled",
        )

        self.glob_opt.pack(side=TOP, fill=X, expand=False)

        self.scroll_canvas = Canvas(master=self.frame_bottom_left)
        self.frame = Frame(master=self.scroll_canvas)
        self.scrollbar = Scrollbar(
            master=self.frame_bottom_left,
            command=self.scroll_canvas.yview,
            orient=VERTICAL,
        )
        self.scroll_canvas.config(yscrollcommand=self.scrollbar.set)

        self.scroll_canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.scroll_canvas.pack(side=LEFT, fill=Y, expand=False)
        self.scrollbar.pack(side=RIGHT, fill=Y, expand=False)

        self.frame_top_left.pack(side=TOP, fill=X, expand=False)
        self.frame_bottom_left.pack(side=BOTTOM, fill=BOTH, expand=True)

        self.frame_left.pack(side=LEFT, fill=Y, expand=False)

        self.canvas = Canvas(master=self.top)

        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.scroll_start)
        self.canvas.bind("<B1-Motion>", self.scroll_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.top.bind_all(
            "<Key-1>",
            lambda _: self._render(opt=True)
            if self.glob_opt["state"] == "normal"
            else None,
        )
        self.top.bind_all("<Control-Key-1>", lambda _: self.gen_prompt(ParityGame))
        self.top.bind_all("<Control-Key-2>", lambda _: self.gen_prompt(MeanPayoffGame))
        self.top.bind_all("<Control-Key-3>", lambda _: self.gen_prompt(EnergyGame))
        self.top.bind_all(
            "<Control-Key-4>", lambda _: self.gen_prompt(DiscountedPayoffGame)
        )
        self.top.bind_all(
            "<Control-Key-5>", lambda _: self.gen_prompt(SimpleStochasticGame)
        )
        self.top.bind_all("<Control-Key-s>", self.save_file_as)
        self.top.bind_all("<Control-Key-o>", self.open_file)

        self.render()

        self.top.mainloop()

    def get_strat(self):

        strat = np.full(len(self.game.owner), -1)

        for i, src in enumerate(self.sources_0):
            tgt = self.tgt_sb_0[i].get()
            if tgt != "":
                strat[src] = int(tgt)

        for i, src in enumerate(self.sources_1):
            tgt = self.tgt_sb_1[i].get()
            if tgt != "":
                strat[src] = int(tgt)

        if self.game.__class__ in [MeanPayoffGame, DiscountedPayoffGame, EnergyGame]:

            mini = np.iinfo(self.game.edges.dtype).min

            for i, (k, l) in enumerate(
                zip(
                    np.count_nonzero(self.game.edges != mini, 1) == 1,
                    np.argmax(self.game.edges, 1),
                )
            ):
                if k:
                    strat[i] = l

        elif self.game.__class__ in [ParityGame, SimpleStochasticGame]:

            for i, (k, l) in enumerate(
                zip(
                    np.count_nonzero(self.game.edges, 1) == 1,
                    np.argmax(self.game.edges, 1),
                )
            ):
                if k:
                    strat[i] = l

        return strat

    def _render(self, restr_values=None, opt=False):

        if hasattr(self, "game"):

            if opt:
                if not hasattr(self, "values"):
                    self.strat = self.game.solve_strat()
                    self.values = self.game.solve_value()

                for i, s in enumerate(self.sources_0):
                    self.tgt_sb_0[i].config(state="normal")
                    self.tgt_sb_0[i].delete(0, END)
                    self.tgt_sb_0[i].insert(0, f"{self.strat[s]}")
                    self.tgt_sb_0[i].config(state="readonly")

                for i, s in enumerate(self.sources_1):
                    self.tgt_sb_1[i].config(state="normal")
                    self.tgt_sb_1[i].delete(0, END)
                    self.tgt_sb_1[i].insert(0, f"{self.strat[s]}")
                    self.tgt_sb_1[i].config(state="readonly")

                values = self.values
                strat = self.strat
                self.glob_opt.config(state="disabled")
            else:
                strat = self.get_strat()
                values = self.game.solve_value(strat=strat)
                self.glob_opt.config(state="normal")

            im = self.game.visualise(strat=strat, values=values, tmp=True)

            self.img = Image.open(im)

            self.img.load()

            remove(im)

            self.img = self.img.resize(
                (
                    int(self.img.size[0] * self.scale),
                    int(self.img.size[1] * self.scale),
                ),
                Image.ANTIALIAS,
            )
            self.pi = ImageTk.PhotoImage(self.img)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.pi)
            self.canvas.config(
                width=min(self.img.width, self.top.winfo_screenwidth() / 2 ** (1 / 2)),
                height=min(
                    self.img.height, self.top.winfo_screenheight() / 2 ** (1 / 2)
                ),
            )

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def render(self):

        if hasattr(self, "game"):

            for w in self.frame.winfo_children():
                w.destroy()

            if self.game.__class__ in [
                MeanPayoffGame,
                DiscountedPayoffGame,
                EnergyGame,
            ]:

                mini = np.iinfo(self.game.edges.dtype).min

                # source of desired player and successor not trivial
                self.sources_0 = np.where(
                    (self.game.owner == 0)
                    & (np.count_nonzero(self.game.edges != mini, 1) > 1)
                )[0]
                self.sources_1 = np.where(
                    (self.game.owner == 1)
                    & (np.count_nonzero(self.game.edges != mini, 1) > 1)
                )[0]

                src_lab_0 = [
                    Label(self.frame, text=f"{src} ↦") for src in self.sources_0
                ]
                src_lab_1 = [
                    Label(self.frame, text=f"{src} ↦") for src in self.sources_1
                ]
                self.tgt_sb_0 = [
                    Spinbox(
                        self.frame,
                        values=tuple(
                            i
                            for i, tgt in enumerate(self.game.edges[src])
                            if tgt != mini
                        ),
                        command=self._render,
                        state="readonly",
                        width=ceil(log10(self.game.owner.shape[0])) + 1,
                        wrap=True,
                    )
                    for src in self.sources_0
                ]
                self.tgt_sb_1 = [
                    Spinbox(
                        self.frame,
                        values=tuple(
                            i
                            for i, tgt in enumerate(self.game.edges[src])
                            if tgt != mini
                        ),
                        command=self._render,
                        state="readonly",
                        width=ceil(log10(self.game.owner.shape[0])) + 1,
                        wrap=True,
                    )
                    for src in self.sources_1
                ]

            elif self.game.__class__ in [ParityGame, SimpleStochasticGame]:

                self.sources_0 = np.where(
                    (self.game.owner == 0) & (np.count_nonzero(self.game.edges, 1) > 1)
                )[0]
                self.sources_1 = np.where(
                    (self.game.owner == 1) & (np.count_nonzero(self.game.edges, 1) > 1)
                )[0]

                src_lab_0 = [
                    Label(self.frame, text=f"{src} ↦") for src in self.sources_0
                ]
                src_lab_1 = [
                    Label(self.frame, text=f"{src} ↦") for src in self.sources_1
                ]
                self.tgt_sb_0 = [
                    Spinbox(
                        self.frame,
                        values=tuple(
                            i for i, tgt in enumerate(self.game.edges[src]) if tgt
                        ),
                        command=self._render,
                        state="readonly",
                        width=ceil(log10(self.game.owner.shape[0])) + 1,
                        wrap=True,
                    )
                    for src in self.sources_0
                ]
                self.tgt_sb_1 = [
                    Spinbox(
                        self.frame,
                        values=tuple(
                            i for i, tgt in enumerate(self.game.edges[src]) if tgt
                        ),
                        command=self._render,
                        state="readonly",
                        width=ceil(log10(self.game.owner.shape[0])) + 1,
                        wrap=True,
                    )
                    for src in self.sources_1
                ]

            if self.game.__class__ in [
                MeanPayoffGame,
                DiscountedPayoffGame,
                EnergyGame,
                SimpleStochasticGame,
            ]:
                lab_0 = Label(self.frame, text="Max player strategy")
            else:
                lab_0 = Label(self.frame, text="Even player strategy")
            lab_0.grid(row=0, column=0, columnspan=2, sticky="nw")

            i = 1

            for j, (src, tgt) in enumerate(zip(src_lab_0, self.tgt_sb_0)):
                src.grid(row=(i + j), column=0, sticky="nw")
                tgt.grid(row=(i + j), column=1, sticky="nw")

            i += len(self.sources_0)

            if self.game.__class__ in [
                MeanPayoffGame,
                DiscountedPayoffGame,
                EnergyGame,
                SimpleStochasticGame,
            ]:
                lab_1 = Label(self.frame, text="Min player strategy")
            else:
                lab_1 = Label(self.frame, text="Odd player strategy")
            lab_1.grid(row=i, column=0, columnspan=2, sticky="nw")

            i += 1

            for j, (src, tgt) in enumerate(zip(src_lab_1, self.tgt_sb_1)):
                src.grid(row=(i + j), column=0, sticky="nw")
                tgt.grid(row=(i + j), column=1, sticky="nw")

            self.glob_opt.config(state="normal")

        self.frame.bind(
            "<Configure>",
            lambda x: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            ),
        )
        self.scroll_canvas.update()
        self.scroll_canvas.config(width=self.scroll_canvas.bbox("all")[2])

        self._render()

    def generate(self, typ, *arg):

        if not 1 < arg[1] <= arg[0]:
            messagebox.showwarning(
                "",
                "Nodes # has to be at least 1\nAvg. outgoing Edges per Node has to be at least 1\nNodes # has to be greater or equal to Avg. outgoing Edges per Node",
            )
            return

        if hasattr(self, "values"):
            del self.values
        if hasattr(self, "strat"):
            del self.strat

        if typ in [MeanPayoffGame, DiscountedPayoffGame, EnergyGame, ParityGame]:
            self.game = typ.generate(arg[0], arg[1] / arg[0], arg[2])
        else:
            self.game = SimpleStochasticGame.generate(arg[0], arg[1] / arg[0])

        self.scale = 1.0

        self.render()

    def gen_prompt(self, typ):

        top = Toplevel()
        top.resizable(False, False)
        top.title(f"Generate {typ.__name__}")
        if typ != SimpleStochasticGame:
            nodes_lab = Label(top, text="Nodes #")
        else:
            nodes_lab = Label(top, text="Nodes #; excl. sinks, avg.")
        nodes_n = Spinbox(top, from_=1, to=100)
        nodes_n.delete(0, 1)
        nodes_n.insert(0, 8)
        out_lab = Label(top, text="Avg. outgoing Edges per Node", anchor="w")
        avg_out = Spinbox(top, from_=1, to=10, increment=0.05)
        avg_out.delete(0, 4)
        avg_out.insert(0, 2)
        if typ != SimpleStochasticGame:
            if typ == ParityGame:
                maxi_lab = Label(top, text="Max. Priority", anchor="w")
            else:
                maxi_lab = Label(top, text="Max. Edge Weight", anchor="w")
            maxi = Spinbox(top, from_=1, to=100, increment=1)
            maxi.delete(0, 4)
            maxi.insert(0, 4)
            button = Button(
                top,
                text="Generate",
                command=lambda: [
                    self.generate(
                        typ, int(nodes_n.get()), float(avg_out.get()), int(maxi.get())
                    ),
                    top.destroy(),
                ],
            )
            button.bind(
                "<Return>",
                lambda _: [
                    self.generate(
                        typ, int(nodes_n.get()), float(avg_out.get()), int(maxi.get())
                    ),
                    top.destroy(),
                ],
            )
        else:
            button = Button(
                top,
                text="Generate",
                command=lambda: [
                    self.generate(typ, int(nodes_n.get()), float(avg_out.get())),
                    top.destroy(),
                ],
            )
            button.bind(
                "<Return>",
                lambda _: [
                    self.generate(typ, int(nodes_n.get()), float(avg_out.get())),
                    top.destroy(),
                ],
            )

        nodes_lab.grid(row=1, column=1, sticky="w")
        out_lab.grid(row=2, column=1, sticky="w")
        nodes_n.grid(row=1, column=2)
        avg_out.grid(row=2, column=2)
        if typ != SimpleStochasticGame:
            maxi_lab.grid(row=3, column=1, sticky="w")
            maxi.grid(row=3, column=2)
        button.grid(row=4, column=1, columnspan=2, sticky="we")

        button.focus()

    def open_file(self, *_):

        file = filedialog.askopenfilename(
            parent=self.top,
            initialdir=join(dirname(realpath(__file__)), "graphs"),
            filetypes=[
                ("csv, binary", ".csv .bin"),
            ],
        )

        if file:
            if hasattr(self, "values"):
                del self.values
            if hasattr(self, "strats"):
                del self.strats
            self.glob_opt.config(state="normal")
            if file.endswith(".bin"):
                self.game = Game.load_bin(file)
            elif file.endswith(".csv"):
                if isfile(file):
                    with open(file, "r") as file_:
                        typee = str(file_.readline().replace("\n", ""))
                    if typee == "pg":
                        self.game = ParityGame.load_csv(file)
                    elif typee == "mpg":
                        self.game = MeanPayoffGame.load_csv(file)
                    elif typee == "eg":
                        self.game = EnergyGame.load_csv(file)
                    elif typee == "dpg":
                        self.game = DiscountedPayoffGame.load_csv(file)
                    elif typee == "ssg":
                        self.game = SimpleStochasticGame.load_csv(file)
            self.scale = 1.0
            self.render()

    def save_file_as(self, *_):

        if hasattr(self, "game"):
            file = filedialog.asksaveasfilename(
                parent=self.top,
                initialdir=join(dirname(realpath(__file__)), "graphs"),
                initialfile=f"{self.game.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
                filetypes=[("csv", ".csv"), ("binary", ".bin")],
                defaultextension=".bin",
            )
            if file:
                if file.endswith(".bin"):
                    self.game.save_bin(file)
                elif file.endswith(".csv"):
                    self.game.save_csv(file)

    def scroll_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, e):

        if (
            (zoom_factor ** -4)
            <= self.scale * (zoom_factor ** (e.delta / 120))
            <= (zoom_factor ** 2)
        ):
            self.scale *= zoom_factor ** (e.delta / 120)
            self._render()


if __name__ == "__main__":

    Gui()
