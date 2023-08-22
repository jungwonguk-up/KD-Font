#ch_sampling_letter_CategorySame
import math, random,copy,os
import numpy as np
seed = 1012

np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class make_sampling_of_pick_letter:
    def __init__(self,candidate_letter_information_list,picking_letter_number,category_number,number_letter_category):
        self.picking_letter_number = picking_letter_number
        self.candidate_letter_information_list = candidate_letter_information_list #[letter, category, stroke]
        self.letter_amount = len(candidate_letter_information_list)
        self.category_number = category_number
        self.number_letter_category = number_letter_category
        self.picked_letters = []
        self.max_category_stroke_count = self.max_set_category_count()

        # stroke 갯수
        self.category_count = [0 for _ in range(8)]
        self.category_stroke_count = [[0 for _ in range(32)] for _ in range(8)]

    def max_set_category_count(self):
        ch_strokes = open("chinese_stroke.txt")
        ch_categorys = open("chinese_category.txt")
        ch_category_stroke_count = [[0 for _ in range(32)] for _ in range(8)]
        ch_max_category_stroke_count = [[0 for _ in range(32)] for _ in range(8)]
        for ch_category, ch_stroke in zip(ch_categorys, ch_strokes):
            ch_stroke = ch_stroke.split()[1:]
            for idx in range(32):
                ch_category_stroke_count[int(ch_category.split()[1])][idx] += int(ch_stroke[idx])
        print("stroke count : ",len(ch_category_stroke_count),len(ch_category_stroke_count[0]))
        for idx in range(len(ch_category_stroke_count)):
            for jdx in range(len(ch_category_stroke_count[0])):
                ch_max_category_stroke_count[idx][jdx] = math.ceil(ch_category_stroke_count[idx][jdx] * ((self.picking_letter_number / self.category_number) / int(self.number_letter_category[idx]) ))

        return ch_max_category_stroke_count
        
    def reset_variable(self):
        self.category_stroke_count = [[0 for _ in range(32)] for _ in range(8)]
        self.picked_letters = []
        self.category_count = [0 for _ in range(8)]

    def check_max_category_count(self,category,stroke):
        if self.category_count[category] > (self.picking_letter_number // self.category_number):
            return False
        for idx in range(32):
            if self.category_stroke_count[category][idx] + int(stroke[idx]) > self.max_category_stroke_count[category][idx]:
                return False
        return True
    
    def letter_random_sampling(self):
        not_pick_list = copy.deepcopy(self.candidate_letter_information_list)
        candidate_list =copy.deepcopy(not_pick_list)
        
        while 1:
            pick_letter_information_list_index = random.choice(range(len(candidate_list)))
            pick_letter_information = candidate_list[pick_letter_information_list_index]
            
            picked_letter = pick_letter_information[0]
            picked_category = int(pick_letter_information[1])
            picked_stroke = pick_letter_information[2]
            
            # check 
            if not self.check_max_category_count(category=picked_category,stroke=picked_stroke):
                candidate_list.pop(pick_letter_information_list_index)
                if len(candidate_list) <= 0:
                    candidate_list = copy.deepcopy(not_pick_list)
                continue
            else:
                for idx in range(32):
                    self.category_stroke_count[picked_category][idx] += int(picked_stroke[idx])
                self.category_count[picked_category] += 1
                self.picked_letters.append(picked_letter)
                not_pick_list.remove(pick_letter_information)
                candidate_list =copy.deepcopy(not_pick_list)
                if self.letter_amount - len(not_pick_list) >= self.picking_letter_number:
                    print(self.picked_letters)
                    print(self.max_category_stroke_count)
                    print(self.category_stroke_count)
                    print(self.category_count)
                    return self.picked_letters
                
if __name__ == "__main__":
    ch_strokes = open("chinese_stroke.txt")
    ch_categorys = open("chinese_category.txt")
    candidate_letter_information_list = []
    number_letter_category = [1564, 2983, 402, 248, 57, 360, 115, 896]
    for ch_category, ch_stroke in zip(ch_categorys, ch_strokes):
        candidate_letter_information = []
        candidate_letter_information.append(ch_category.split()[0])
        candidate_letter_information.append(ch_category.split()[1])
        candidate_letter_information.append(ch_stroke.split()[1:])
        candidate_letter_information_list.append(candidate_letter_information)
        

    pick_letter = make_sampling_of_pick_letter(candidate_letter_information_list=candidate_letter_information_list,picking_letter_number=420,category_number=8,number_letter_category=number_letter_category)
    pick_letter.letter_random_sampling()