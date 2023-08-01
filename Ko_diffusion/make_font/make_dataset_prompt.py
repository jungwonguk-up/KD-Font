from tkinter import *
import os
from PIL import Image, ImageDraw, ImageFont,ImageTk

IVORY = "#FFE4C0"
PINK = "#FFBBBB"
BLUE = "#BFFFF0"
GREEN = "#BFFFF0"
BLACK = "#000000"
BG_COLOR = "#21325E"
CORRECT_COLOR = "#F1D00A"
BTN_COLOR = "#F0F0F0"

def make_letter_image(image_size, letter):
    font = ImageFont.truetype(font="MaShanZheng-Regular.ttf", size=int(image_size*0.8))
    x,y = font.getsize(letter)
    letter_image = Image.new('RGB', (image_size,image_size), color='white')
    draw_image = ImageDraw.Draw(letter_image)
    draw_image.text(((image_size-x)/2, (image_size-y)/2), letter, font=font, fill='black')
    return letter_image

    # def dispaly_letter_image(self,letter_image):
    #     letter_image = ImageTk.PhotoImage(letter_image)
    #     label = Label(self.window,image=letter_image)
    #     label.image=letter_image
    #     # label.pack(pady=10) 
    #     label.grid(column=1,column=0)


class windows_tkinter:
    def __init__(self,start_number,ch_letters_sum,image_size=100):
        self.window = Tk()
        self.start_number= start_number
        self.current_number = start_number
        self.ch_letters_sum = ch_letters_sum
        self.image_size = image_size
        
        self.class_width = 0.075
        self.class_height = 0.075
        self.class_font_size = 0.1
        self.class_relx = 0.01
        self.class_rely = 0.45
        self.class_move_ratio = 0.08


        self.second_class_images= [None for _ in range(32)]
        self.second_class_buttons = [None for _ in range(32)]
        self.category_buttons = [None for _ in range(5)]
        self.class_buttons = [None for _ in range(11)]
        
        # clear method
        self.second_class_value_info = [None for _ in range(32)]
        self.second_class = [[0 for _ in range(32)] for _ in range(11)]
        self.click_category = 100
        self.click_class = 100
        self.click_second_class = 100
        self.second_class_click_number = 0
        self.second_class_value_info_on = False
        self.letter_image_label = None
        self.counting_label = None
    
    def clear_button_action(self):
        for scbdx in range(32):
            self.second_class_value_info[scbdx]['text'] = 0
            self.second_class_value_info[scbdx].place_forget()
        self.second_class_value_info_on = False

        self.click_category = 100
        self.click_class = 100
        self.click_second_class = 100
        self.second_class_click_number = 0
        self.second_class = [[0 for _ in range(32)] for _ in range(11)]

    def next_button_action(self):
        with open('ch_label.txt','a',encoding='utf-8') as f:
            letter_embedding = f'{self.ch_letters_sum[self.current_number][0]} {self.second_class}\n'
            f.write(letter_embedding)
        
        self.clear_button_action()
        self.current_number += 1
        letter_img = make_letter_image(self.image_size*2,self.ch_letters_sum[self.current_number][0])
        letter_image = ImageTk.PhotoImage(letter_img)
        self.letter_image_label['image'] = letter_image
        self.letter_image_label.image=letter_image
        
        
    def forget_class_buttons(self):
        for idx in range(11):
            if self.class_buttons[idx] is not None:
                self.class_buttons[idx].place_forget()

    def display_second_class_value_info_first(self,):
        for scbdx in range(32):
            self.second_class_value_info[scbdx].place(relx = 0.25 + (scbdx % 8) * 0.09,rely=0.6+(scbdx //8)*0.09)
        print('first')

    def update_second_class_value_info_WSCC(self,scdx):
        self.click_second_class = scdx
        self.second_class[self.click_class][self.click_second_class] += 1
        self.second_class_value_info[self.click_second_class]['text'] = self.second_class[self.click_class][self.click_second_class]
        self.second_class_click_number +=1
        self.counting_label['text'] = f"{self.second_class_click_number}획"

    def display_second_class_value_info_WCC(self,cdx):
        self.click_class = cdx
        
        if self.second_class_value_info_on == False: 
            self.display_second_class_value_info_first()
            self.second_class_value_info_on = True

        for cdx in range(11):
            if self.class_buttons[cdx]['bg'] is not BTN_COLOR:
                self.class_buttons[cdx]['bg'] = BTN_COLOR
        
        self.class_buttons[self.click_class]['bg']=PINK
        
        for scvdx in range(32):
            self.second_class_value_info[scvdx]['text'] = self.second_class[self.click_class][scvdx]

        
    # def clear
    

    def place_class_buttons(self, idx):
        self.click_category = idx
        self.forget_class_buttons()
        
        if idx == 0:
            self.class_buttons[0].place(relx =self.class_relx,rely=self.class_rely)
            
        elif idx == 1:
            for ydx in range(1,3):
                self.class_buttons[ydx].place(relx =self.class_relx+(ydx-1)*self.class_move_ratio,rely=self.class_rely)

        elif idx == 2:
            for ydx in range(3,5):
                self.class_buttons[ydx].place(relx =self.class_relx+(ydx-3)*self.class_move_ratio,rely=self.class_rely)

        elif idx == 3:
            for ydx in range(5,8):
                self.class_buttons[ydx].place(relx =self.class_relx+(ydx-5)*self.class_move_ratio,rely=self.class_rely)

        elif idx == 4:
            for ydx in range(8,11):
                self.class_buttons[ydx].place(relx =self.class_relx+(ydx-8)*self.class_move_ratio,rely=self.class_rely)

    def display_window(self):
        ch_label_path = './ch_image'
        ch_label_image_name = os.listdir(ch_label_path)
        self.window.title("중국어 데이터셋 제작 툴")
        self.window.resizable(False,False)
        self.window.geometry(f"{self.image_size*10}x{self.image_size*10}")
        # self.window.config(padx=10,pady=10,bg=BG_COLOR)
        self.window.config(bg=BG_COLOR)

        entry = Entry(self.window,font=('나눔바른펜',20),bg=BG_COLOR)



        
        letter_img = make_letter_image(self.image_size*2,self.ch_letters_sum[self.start_number][0])
        letter_image = ImageTk.PhotoImage(letter_img)
        self.letter_image_label = Label(self.window,image=letter_image)
        self.letter_image_label.image=letter_image
        self.letter_image_label.place(relx=0.02,rely=0.21,relwidth=0.2)
        
        self.goal_counting_label = Label(self.window,text=f"{self.ch_letters_sum[self.current_number][1]}획",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR)
        self.goal_counting_label.place(relx=0.02,rely=0.65,relwidth=0.2,relheight=0.06)
        self.counting_label = Label(self.window,text=f"0획",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR)
        self.counting_label.place(relx=0.02,rely=0.75,relwidth=0.2,relheight=0.06)
        
        next_button = Button(self.window,text=f"NEXT",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.next_button_action())
        next_button.place(relx=0.02,rely=0.02,relwidth=0.2,relheight=0.06)
        clear_button = Button(self.window,text=f"CLEAR",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.clear_button_action())
        clear_button.place(relx=0.02,rely=0.1,relwidth=0.2,relheight=0.06)



    
        for idx in range(5):
            self.category_buttons[idx] = Button(self.window,text=f"{idx+1}번 Category",width = int(self.image_size*0.15),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda ccdx = idx:self.place_class_buttons(ccdx))
            self.category_buttons[idx].place(relx = 0.27+0.14*(idx),rely=0.02)

        for cdx in range(11):
            self.class_buttons[cdx] = Button(self.window,text=f"{cdx+1}번 class",width = int(self.image_size*self.class_width),height=int(self.image_size*self.class_height),font=('나눔바른펜',int(self.image_size*self.class_font_size)),bg=BTN_COLOR,command=lambda ccdx = cdx:self.display_second_class_value_info_WCC(ccdx))
        
        for scdx in range(32):
            self.second_class_images[scdx] = ImageTk.PhotoImage(Image.open(os.path.join(ch_label_path,ch_label_image_name[scdx])).resize((int(self.image_size*0.8),int(self.image_size*0.8)), Image.ANTIALIAS))
            self.second_class_buttons[scdx] = Button(self.window,image=self.second_class_images[scdx],command=lambda svdx = scdx : self.update_second_class_value_info_WSCC(svdx))
            self.second_class_buttons[scdx].place(relx = 0.25 + (scdx % 8) * 0.09,rely=0.2+(scdx //8)*0.09)
        for scbdx in range(32):
            self.second_class_value_info[scbdx] = Label(self.window,text=0,width = int(self.image_size*0.1),height=int(self.image_size*0.08),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR)
        self.window.mainloop()



if __name__ == "__main__":
    ch_letters_sum = []
    with open('ch_letter_sum.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            ch_letters_sum.append(line.split())
    # print(ch_letters_sum)
    ss = windows_tkinter(start_number=1000,ch_letters_sum=ch_letters_sum,image_size=100)
    ss.display_window()


