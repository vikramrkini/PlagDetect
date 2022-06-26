from tkinter import Button,Tk,Label,StringVar
from tkinter.font import Font,BOLD
from types import new_class
from siamese_network import get_siamese_sim
from os.path import isfile ,basename
from tkinter import messagebox as mb
from tkinter.filedialog import askopenfilename
import tensorflow as tf

s = tf.qint16

WIDTH , HEIGHT = 800, 550
centerx , centery = WIDTH//2 , HEIGHT// 2
bg_color = "#a4d5eb"

siamese_sim = get_siamese_sim()


def get_root():
    """Prepare the root to be ready"""
    root = Tk()
    root.title("Plag Detect")
    root['background']='#856ff8'
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width/2) - (WIDTH/2)
    y = (screen_height/2) - (HEIGHT/2)
    root.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, x, y))
    root.resizable(False, False)
    return root


root = get_root()
src_file = StringVar(root , "None")
sus_file = StringVar(root ,  "None")
plag_score_var = StringVar(root ,  "Plag Score:")


global src_file_path , sus_file_path
src_file_path = sus_file_path = None

def get_src_path():
    global src_file_path
    src_file_path = askopenfilename(filetypes=(
        ("Text file", "*.txt"), 
        ("All files", "*.*"))
    )
    print(src_file_path)

    src_file.set(f"{basename(src_file_path)}")

def get_sus_path():
    global sus_file_path
    sus_file_path = askopenfilename(filetypes=(
        ("Text file", "*.txt"), 
        ("All files", "*.*"))
    )

    print(sus_file_path)

    sus_file.set(f"{basename(sus_file_path)}")


def submit_cmd():
    # src_path = src_file.get()
    # sus_path = sus_file.get()

    if isfile(src_file_path) and isfile(sus_file_path):
        with open(src_file_path ,'r',encoding = 'utf8') as f:
            src_text = f.read()

        with open(sus_file_path ,'r',encoding = 'utf8') as f:
            sus_text = f.read()

        # new_score = 0.2
        new_score = siamese_sim(src_text , sus_text)
        new_score = round( float(new_score) , 4)
        plag_score_var.set( f"Plag Score : { new_score }")
    else:
        mb.showerror("Invalid path." , "Please provide correct file path.!")


def render_page():
    seed_x , seed_y = 280 , centery - 200
    offset = 200

    choose_src_btn = Button(
                    root, 
                    text= "Choose Source File",
                    font= Font(root, size=20, weight=BOLD),
                    anchor='center',
                    command = get_src_path
            )

    choose_susp_btn = Button(
                root, 
                text= "Choose Susp File",
                font= Font(root, size=20, weight=BOLD),
                anchor='center',
                command = get_sus_path
        )

    src_file_label = Label(
                  root, 
                  text='',
                  font=Font(root, size=12, weight=BOLD),
                  textvariable = src_file,
            )

    sus_file_label = Label(
                  root, 
                  text='',
                  font=Font(root, size=12, weight=BOLD),
                  textvariable = sus_file,

            )

    plag_score_label = Label(
                  root, 
                  text='Plag Score',
                  font=Font(root, size=22, weight=BOLD),
                  textvariable = plag_score_var,
            )

    submit_btn = Button(
                root, 
                text= "Submit",
                font= Font(root, size=20, weight=BOLD),
                command = submit_cmd,
        )


    choose_src_btn.place(x = seed_x - offset , y = seed_y)
    choose_susp_btn.place(x = seed_x + offset , y = seed_y)

    seed_y += 70
    src_file_label.place(x = seed_x - 120, y = seed_y)
    sus_file_label.place(x = seed_x + 210 , y = seed_y)

    seed_y += 100
    submit_btn.place(x = centerx - 40 , y = seed_y )

    seed_y += 150
    plag_score_label.place(x = centerx - 100 , y = seed_y)


if __name__ == "__main__":
    render_page()
    root.mainloop()
   
    pass