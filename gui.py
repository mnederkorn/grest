from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from grest import *
from math import ceil, log

class Gui:

    def __init__(self):

        self.top = Tk()

        self.menubar = Menu(self.top)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.newmenu = Menu(self.filemenu, tearoff=0)
        self.newmenu.add_command(label="Empty", accelerator="Ctrl-N")
        self.newmenu.add_command(label="Generate", accelerator="Ctrl-G")
        self.filemenu.add_cascade(label="New", menu=self.newmenu)
        self.filemenu.add_command(label="Open", accelerator="Ctrl-O")
        self.filemenu.add_command(label="Save", accelerator="Ctrl-S")
        self.filemenu.add_command(label="Save as", accelerator="Ctrl-Shift-S")
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.top.config(menu=self.menubar)

        self.frame_left = Frame(self.top)

        self.scroll_canvas = Canvas(master=self.frame_left)
        self.frame = Frame(master=self.scroll_canvas)
        self.scroll = Scrollbar(master=self.frame_left, command=self.scroll_canvas.yview, orient=VERTICAL)
        self.scroll_canvas.config(yscrollcommand=self.scroll.set)

        self.scroll_canvas.create_window((0,0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", lambda x: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        mini = np.iinfo(g.edges.dtype).min

        self.sources = [Label(self.frame, text=f"{i} ↦") for i,x in enumerate((~g.owner) & (np.count_nonzero(g.edges!=mini,1)>1)) if x]
        self.srcs = [i for i,x in enumerate((~g.owner) & (np.count_nonzero(g.edges!=mini,1)>1)) if x]
        self.targets = [Spinbox(self.frame, values=("",)+tuple(j for j,y in enumerate(g.edges[i]) if y != mini), command=self.visualise, state="readonly", width=ceil(log(g.owner.shape[0],10)), wrap=True) for i,x in enumerate((~g.owner) & (np.count_nonzero(g.edges!=mini,1)>1)) if x]

        self.scroll_canvas.pack(side=LEFT, fill=Y, expand=False)
        self.scroll.pack(side=RIGHT, fill=Y, expand=False)

        for i,x in enumerate(self.sources):
            self.sources[i].grid(row=i+1,column=0, sticky="nswe")
            self.targets[i].grid(row=i+1,column=1, sticky="nswe")

        self.frame_left.pack(side=LEFT, fill=Y, expand=False)

        self.canvas = Canvas(master=self.top)

        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.scroll_start)
        self.canvas.bind("<B1-Motion>", self.scroll_move)

        self.scroll_canvas.update()
        self.scroll_canvas.config(width=self.scroll_canvas.bbox("all")[2])

        self.visualise()

        self.top.mainloop()

    def scroll_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def visualise(self):

        # strat = np.empty((0,2), dtype=np.int32)
        strat = np.full(len(g.owner),-1)

        for src,tgt in zip(self.srcs, self.targets):
            _tgt = tgt.get()
            if _tgt != "":
                strat[src]=int(_tgt)

        mini = np.iinfo(g.edges.dtype).min

        for i,(k,l) in enumerate(zip(np.count_nonzero(g.edges!=mini,1)==1,np.argmax(g.edges,1))):
            if k:
                strat[i]=l

        im = g.visualise(strat=strat)

        self.img = Image.open(im)
        self.img.load()
        self.pi = ImageTk.PhotoImage(self.img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.pi)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.config(width=min(self.img.width, self.top.winfo_screenwidth()/2**(1/2)), height=min(self.img.height, self.top.winfo_screenheight()/2**(1/2)))

if __name__ == '__main__':

    g=Game.load(r"C:\ata\uni\master\grest\grest\test.bin")

    Gui()