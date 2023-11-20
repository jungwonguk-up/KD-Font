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
    font = ImageFont.truetype(font="SIMSUN.ttf", size=int(image_size*0.8))
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
        self.category_amount = 8
        self.click_category = 100
        self.category_buttons = [None for _ in range(self.category_amount)]
        self.selected_category_label = None
        self.category_images, self.selected_category_images = self.get_category_image()
    
    def get_category_image(self):
        category_images = []
        selected_category_images = []
        for idx in range(self.category_amount):
            img = Image.open(f"./category_image/category_{idx}.png")
            resized_img = img.resize((int(self.image_size*2),int(self.image_size*1)))
            resized_img= ImageTk.PhotoImage(resized_img)

            selected_resized_img = img.resize((int(self.image_size*3.80),int(self.image_size*1.95)))
            selected_resized_img = ImageTk.PhotoImage(selected_resized_img)

            category_images.append(resized_img)
            selected_category_images.append(selected_resized_img)
        return category_images, selected_category_images
    def clear_button_action(self):
        self.click_category = 100
        self.selected_category_image['image'] = self.selected_category_images[0]
        self.selected_category_label['text'] = "None Selected"
    def next_button_action(self):
        if self.click_category != 100:
            with open('ch_category.txt','a',encoding='utf-8') as f:
                letter_embedding = f'{self.ch_letters_sum[self.current_number][0]} {self.click_category}\n'
                f.write(letter_embedding)
        
        self.clear_button_action()
        self.current_number += 1
        letter_img = make_letter_image(self.image_size*2,self.ch_letters_sum[self.current_number][0])
        letter_image = ImageTk.PhotoImage(letter_img)
        self.letter_image_label['image'] = letter_image
        self.letter_image_label.image=letter_image
        self.goal_counting_label['text'] = f"{self.ch_letters_sum[self.current_number][1]}획"
        
    

    def button_click_category(self,cdx):
        self.click_category = cdx
        print(self.click_category)
        self.selected_category_label['text'] = f'{cdx} Category'
        self.selected_category_image['image'] = self.selected_category_images[cdx]
        

    def display_window(self):
        self.window.title("중국어 데이터셋 제작 툴")
        self.window.resizable(False,False)
        self.window.geometry(f"{self.image_size*10}x{self.image_size*10}")
        # self.window.config(padx=10,pady=10,bg=BG_COLOR)
        self.window.config(bg=BG_COLOR)

        letter_img = make_letter_image(self.image_size*2,self.ch_letters_sum[self.start_number][0])
        letter_image = ImageTk.PhotoImage(letter_img)
        self.letter_image_label = Label(self.window,image=letter_image)
        self.letter_image_label.image=letter_image
        self.letter_image_label.place(relx=0.02,rely=0.21,relwidth=0.2)
        
        next_button = Button(self.window,text=f"NEXT",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.next_button_action())
        next_button.place(relx=0.02,rely=0.02,relwidth=0.2,relheight=0.06)
        clear_button = Button(self.window,text=f"CLEAR",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.clear_button_action())
        clear_button.place(relx=0.02,rely=0.1,relwidth=0.2,relheight=0.06)
        
        selected_label = Label(self.window,text=f"Selected Category",width = int(self.image_size*1.1),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR)
        selected_label.place(relx=0.05,rely=0.5)

        self.selected_category_label = Label(self.window,text=f"None Selected",width = int(self.image_size*0.5),height=int(self.image_size*0.15),font=('나눔바른펜',int(self.image_size*0.10)),bg=BTN_COLOR)
        self.selected_category_label.place(relx=0.05,rely=0.7)

        self.selected_category_image = Label(self.window,image = self.selected_category_images[0])
        self.selected_category_image.place(relx=0.55,rely=0.7)
        
        for idx in range(self.category_amount):
            self.category_buttons[idx] = Button(self.window,image = self.category_images[idx],command=lambda ccdx = idx:self.button_click_category(ccdx))
            self.category_buttons[idx].place(relx = 0.27+0.24*(idx%3),rely=0.02 + 0.15*(idx//3))

        

        self.window.mainloop()



if __name__ == "__main__":
    ch_letters_sum = []
    with open('ch_letter_sum.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            ch_letters_sum.append(line.split())
    # print(ch_letters_sum)
    ss = windows_tkinter(start_number=4695,ch_letters_sum=ch_letters_sum,image_size=100)
    ss.display_window()


