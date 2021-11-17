from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from math import ceil, log10

from grest import *
from pg import ParityGame
from mpg import MeanPayoffGame
from dpg import DiscountedPayoffGame
from ssg import SimpleStochasticGame
from eg import EnergyGame

zoom_factor = (2**(1/2))

class Gui:

    def __init__(self):

        self.top = Tk()

        self.menubar = Menu(self.top)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.gen_menu = Menu(self.filemenu, tearoff=0)
        self.gen_menu.add_command(label="PG", command=lambda: self.gen_prompt("PG"), accelerator="Ctrl-1")
        self.gen_menu.add_command(label="MPG", command=lambda: self.gen_prompt("MPG"), accelerator="Ctrl-2")
        self.gen_menu.add_command(label="DPG", command=lambda: self.gen_prompt("DPG"), accelerator="Ctrl-3")
        self.gen_menu.add_command(label="EG", command=lambda: self.gen_prompt("EG"), accelerator="Ctrl-4")
        self.gen_menu.add_command(label="SSG", command=lambda: self.gen_prompt("SSG"), accelerator="Ctrl-5")
        self.filemenu.add_cascade(label="Generate", menu=self.gen_menu)
        self.filemenu.add_command(label="Open", accelerator="Ctrl-O")
        self.filemenu.add_command(label="Save", accelerator="Ctrl-S")
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.top.config(menu=self.menubar)

        self.frame_left = Frame(self.top)

        self.scale = 1.0

        self.scroll_canvas = Canvas(master=self.frame_left)
        self.frame = Frame(master=self.scroll_canvas)
        self.scrollbar = Scrollbar(master=self.frame_left, command=self.scroll_canvas.yview, orient=VERTICAL)
        self.scroll_canvas.config(yscrollcommand=self.scrollbar.set)

        self.scroll_canvas.create_window((0,0), window=self.frame, anchor="nw")

        self.scroll_canvas.pack(side=LEFT, fill=Y, expand=False)
        self.scrollbar.pack(side=RIGHT, fill=Y, expand=False)

        self.frame_left.pack(side=LEFT, fill=Y, expand=False)

        self.canvas = Canvas(master=self.top)

        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.scroll_start)
        self.canvas.bind("<B1-Motion>", self.scroll_move)
        self.canvas.bind("<MouseWheel>",self.zoom)
        self.top.bind_all("<Control-Key-1>", lambda _: self.gen_prompt("PG"))
        self.top.bind_all("<Control-Key-2>", lambda _: self.gen_prompt("MPG"))
        self.top.bind_all("<Control-Key-3>", lambda _: self.gen_prompt("DPG"))
        self.top.bind_all("<Control-Key-4>", lambda _: self.gen_prompt("EG"))
        self.top.bind_all("<Control-Key-5>", lambda _: self.gen_prompt("SSG"))

        self.render()

        self.top.mainloop()

    def _render(self):

        if hasattr(self, "game"):

            strat = np.full(len(self.game.owner),-1)

            for i,src in enumerate(self.sources):
                tgt = self.tgt_sb[i].get()
                if tgt!="":
                    strat[src]=int(tgt)

            if self.typ=="MPG" or self.typ=="DPG" or self.typ=="EG":

                mini = np.iinfo(self.game.edges.dtype).min

                for i,(k,l) in enumerate(zip(np.count_nonzero(self.game.edges!=mini,1)==1,np.argmax(self.game.edges,1))):
                    if k:
                        strat[i]=l

            elif self.typ=="PG" or self.typ=="SSG":

                for i,(k,l) in enumerate(zip(np.count_nonzero(self.game.edges,1)==1,np.argmax(self.game.edges,1))):
                    if k:
                        strat[i]=l

            im = self.game.visualise(strat=strat)

            self.img = Image.open(im)
            self.img.load()
            self.img=self.img.resize((int(self.img.size[0]*self.scale),int(self.img.size[1]*self.scale)),Image.ANTIALIAS)
            self.pi = ImageTk.PhotoImage(self.img)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.pi)
            self.canvas.config(width=min(self.img.width, self.top.winfo_screenwidth()/2**(1/2)), height=min(self.img.height, self.top.winfo_screenheight()/2**(1/2)))

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def render(self):

        if hasattr(self, "game"):

            if self.typ=="MPG" or self.typ=="DPG" or self.typ=="EG":

                mini = np.iinfo(self.game.edges.dtype).min

                # source is of desired player and not trivial (only one successor)
                self.sources = np.where((self.game.owner==False) & (np.count_nonzero(self.game.edges!=mini,1)>1))[0]

                src_lab = [Label(self.frame, text=f"{src} ↦") for src in self.sources]
                self.tgt_sb = [Spinbox(self.frame, values=("",)+tuple(i for i,tgt in enumerate(self.game.edges[src]) if tgt!=mini), command=self._render, state="readonly", width=ceil(log10(self.game.owner.shape[0]))+1, wrap=True) for src in self.sources]

            elif self.typ=="PG" or self.typ=="SSG":

                self.sources = np.where((self.game.owner==0) & (np.count_nonzero(self.game.edges,1)>1))[0]

                src_lab = [Label(self.frame, text=f"{src} ↦") for src in self.sources]
                self.tgt_sb = [Spinbox(self.frame, values=("",)+tuple(i for i,tgt in enumerate(self.game.edges[src]) if tgt), command=self._render, state="readonly", width=ceil(log10(self.game.owner.shape[0]))+1, wrap=True) for src in self.sources]

            for i,(src,tgt) in enumerate(zip(src_lab,self.tgt_sb)):
                src.grid(row=i,column=0, sticky="nswe")
                tgt.grid(row=i,column=1, sticky="nswe")

        self.frame.bind("<Configure>", lambda x: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.update()
        self.scroll_canvas.config(width=self.scroll_canvas.bbox("all")[2])

        self._render()

    def generate(self, typ, *arg):

        self.typ=typ

        p=arg[1]/arg[0]

        if typ=="PG":
            self.game = ParityGame.generate(arg[0], p, arg[2])
        elif typ=="MPG":
            self.game = MeanPayoffGame.generate(arg[0], p, arg[2])
        elif typ=="DPG":
            self.game = DiscountedPayoffGame.generate(arg[0], p, arg[2])
        elif typ=="EG":
            self.game = EnergyGame.generate(arg[0], p, arg[2])
        elif typ=="SSG":
            self.game = SimpleStochasticGame.generate(arg[0], p)

        self.render()

    def gen_prompt(self, typ):

        top = Toplevel()
        top.resizable(False, False)
        top.title(f"Generate {typ}")
        nodes_lab = Label(top, text="Nodes #")
        nodes_n = Spinbox(top, from_=1, to=100)
        nodes_n.delete(0, 1)
        nodes_n.insert(0, 8)
        out_lab = Label(top, text="Avg. outgoing Edges per Node", anchor="w")
        avg_out = Spinbox(top, from_=1, to=10, increment=0.05)
        avg_out.delete(0, 4)
        avg_out.insert(0, 2)
        if typ!="SSG":
            if typ=="PG":
                maxi_lab = Label(top, text="Max. Priority", anchor="w")
            else:
                maxi_lab = Label(top, text="Max. Edge Weight", anchor="w")
            maxi = Spinbox(top, from_=1, to=100, increment=1)
            maxi.delete(0, 4)
            maxi.insert(0, 4)
            button = Button(top, text="Generate", command=lambda: [self.generate(typ, int(nodes_n.get()), float(avg_out.get()), int(maxi.get())), top.destroy()])
            button.bind("<Return>", lambda _: [self.generate(typ, int(nodes_n.get()), float(avg_out.get()), int(maxi.get())), top.destroy()])
        else:
            button = Button(top, text="Generate", command=lambda: [self.generate(typ, int(nodes_n.get()), float(avg_out.get())), top.destroy()])
            button.bind("<Return>", lambda _: [self.generate(typ, int(nodes_n.get()), float(avg_out.get())), top.destroy()])

        nodes_lab.grid(row=1,column=1,sticky="w")
        out_lab.grid(row=2,column=1,sticky="w")
        nodes_n.grid(row=1,column=2)
        avg_out.grid(row=2,column=2)
        if typ!="SSG":
            maxi_lab.grid(row=3,column=1,sticky="w")       
            maxi.grid(row=3,column=2)            
        button.grid(row=4,column=1,columnspan=2,sticky="we")

        button.focus()

    def scroll_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, e):

        if (zoom_factor**-4)<=self.scale*(zoom_factor**(e.delta/120))<=(zoom_factor**2):
            self.scale*=(zoom_factor**(e.delta/120))
            self._render()

if __name__ == '__main__':

    Gui()