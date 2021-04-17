# -*- coding: utf-8 -*-

from tkinter import *
import tkinter.filedialog
       
from FaceDetectRetrieve_new import *
def main(): 
    root = Tk()
    
    # optionally give it a title
    root.title("Image Retrieval")
    ## set the root window's height, width and x,y position
    ## x and y are the coordinates of the upper left corner
    w = 300
    h = 200
    x = 100
    y = 100
    # use width x height + x_offset + y_offset (no spaces!)
    root.geometry("%dx%d+%d+%d" % (w, h, x, y))
    #
    #label = Label(root, text="Query Image")
    #label.place(x=20, y=30)

     
    QueryImagePath= StringVar() 
    SearchSetPath = StringVar() 
    def Retrieve():
        detectRetrieve(QueryImagePath.get(),SearchSetPath.get(),UserName.get(),NumImages=NumImages.get())
     
    def QueryPath():
        f = tkinter.filedialog.askdirectory(
            parent=root, initialdir='./',
            title='Select Reference Image Folder',
    #        filetypes=[('jpg images', '.jpg'),
    #                   ('gif images', '.gif')]
            )
        
        QueryImagePath.set(f)
     
        print(f)
    
    def SearchSet():
        f = tkinter.filedialog.askdirectory(
            parent=root, initialdir='./',
            title='Choose Events/Games directory',
            )
        
        SearchSetPath.set(f)    
        print(f)
    
    row = 5
    UserName = StringVar()
    
    L1 = Label(root, text="User Name")
    #L1.pack( side = LEFT)
    L1.grid(row = row,column=2)
    E1 = Entry(root, bd =5, textvariable = UserName)
    E1.grid(row = row, column = 3)
    #UserName = E1.get()
    row+=1
    
    L2 = Label(root, text="Find Me")
    L2.grid(row = row, column = 2 )
    E2 = Entry(root, bd =5,textvariable = QueryImagePath)
    E2.grid(row = row , column = 3)
    b1 = Button(root, text='Browse', command=QueryPath)
    b1.grid(row = row , column = 4)
    row +=1
    
    L3 = Label(root, text="Events/Games")
    L3.grid(row = row , column = 2)
    E3 = Entry(root, bd = 5, textvariable = SearchSetPath)
    E3.grid(row = row , column = 3)
    b2 = Button(root, text='Browse', command=SearchSet)
    b2.grid(row = row , column = 4)
    row +=1
    
    L5 = Label(root, text="No of images")
    #L1.pack( side = LEFT)
    L5.grid(row = row,column=2)
    NumImages = StringVar()
    E4 = Entry(root, bd =5, textvariable=NumImages)
    E4.grid(row = row, column = 3)
    #CutOff = E1.get()
    #print("Cutoff",CutOff.get())
    row += 1
    
    L4 = Label(root, text="")
    L4.grid(row = row , column = 2)
    row +=2
    
    #entry = tkinter.Entry(root, textvariable = QueryImagePath).pack()
    RetrieveButton = tkinter.Button(root, text='Retrieve', command=Retrieve)
    RetrieveButton.grid(row = row , column = 3)
    exitButton = Button(root, text='Exit', command=root.destroy)
    exitButton.grid(row = row , column = 4) 
    
    root.mainloop()


if __name__ == '__main__':
    main()