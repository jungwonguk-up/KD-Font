import os
import random,math
import shutil
import numpy as np


class make_sampling_data:
    def __init__(self,max_consonant_letter_count = 10, criteria_consonant_letter = "last"):
        self.max_consonant_letter_count = max_consonant_letter_count
        self.criteria_consonant_letter = criteria_consonant_letter

        # 초성, 중성 종성의 개수
        self.first_consonant_letter_number = 19
        self.middle_consonant_letter_number = 21
        self.last_consonant_letter_number = 28

        # 초성, 중성, 종성 각각의 문자가 사용된 개수 리스트
        self.first_consonant_letters_count = [0 for _ in range(self.first_consonant_letter_number)]  # ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
        self.middle_consonant_letters_count = [0 for _ in range(self.middle_consonant_letter_number)]  # ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ
        self.last_consonant_letters_count = [0 for _ in range(self.last_consonant_letter_number)]  # blank's index is zero # 'blank' + ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ
        self.pick_letter_list = []

    def get_letter_by_unicode(self,index_list):
        return chr((index_list[0] * 588) + (index_list[1] * 28) + (index_list[2]) + 44032)
    
    def check_data_is_representative(self,max_l2nrom):
        # print(np.std(self.first_consonant_letters_count) * np.sqrt(
        #     len(self.first_consonant_letters_count)) / self.max_consonant_letter_count)
        # print(np.std(self.middle_consonant_letters_count) * np.sqrt(
        #     len(self.middle_consonant_letters_count)) / self.max_consonant_letter_count)
        # print(np.std(self.last_consonant_letters_count) * np.sqrt(
        #     len(self.last_consonant_letters_count)) / self.max_consonant_letter_count)
        if len(self.pick_letter_list) == len(set(self.pick_letter_list)):
            if np.std(self.first_consonant_letters_count)*np.sqrt(len(self.first_consonant_letters_count))/self.max_consonant_letter_count <= max_l2nrom:
                if np.std(self.middle_consonant_letters_count)*np.sqrt(len(self.middle_consonant_letters_count))/self.max_consonant_letter_count <= max_l2nrom:
                    if np.std(self.last_consonant_letters_count)*np.sqrt(len(self.last_consonant_letters_count))/self.max_consonant_letter_count <= max_l2nrom:
                        if not self.first_consonant_letters_count.__contains__(0):
                            if not self.middle_consonant_letters_count.__contains__(0):
                                if not self.last_consonant_letters_count.__contains__(0):
                                    return True

        return False

    def reset_variable(self):
        self.first_consonant_letters_count = [0 for _ in range(self.first_consonant_letter_number)]  # ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
        self.middle_consonant_letters_count = [0 for _ in range(self.middle_consonant_letter_number)]  # ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ
        self.last_consonant_letters_count = [0 for _ in range(self.last_consonant_letter_number)]  # blank's index is zero # 'blank' + ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ
        self.pick_letter_list = []
    
    # criteria_consonant_letter's factor is first, middle, last
    def letter_random_sampling(self):
        while 1:
            # pick_unicode_list = []
            if self.criteria_consonant_letter == "last":
                max_first_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.last_consonant_letter_number / self.first_consonant_letter_number )
                max_middle_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.last_consonant_letter_number / self.middle_consonant_letter_number)
                for last_consonant_letter_index in range(self.last_consonant_letter_number):
                    while self.last_consonant_letters_count[last_consonant_letter_index] < self.max_consonant_letter_count:
                        first_consonant_letter_index =  random.randrange(0,self.first_consonant_letter_number)
                        middle_consonant_letter_index =  random.randrange(0,self.middle_consonant_letter_number)
                        if self.first_consonant_letters_count[first_consonant_letter_index] < max_first_consonant_letter_count and self.middle_consonant_letters_count[middle_consonant_letter_index] < max_middle_consonant_letter_count:
                            self.pick_letter_list.append(self.get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                            # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                            self.first_consonant_letters_count[first_consonant_letter_index] += 1
                            self.middle_consonant_letters_count[middle_consonant_letter_index] += 1
                            self.last_consonant_letters_count[last_consonant_letter_index] += 1

            elif self.criteria_consonant_letter == "middle":
                max_first_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.middle_consonant_letter_number / self.first_consonant_letter_number )
                max_last_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.middle_consonant_letter_number / self.last_consonant_letter_number)
                for middle_consonant_letter_index in range(self.middle_consonant_letter_number):
                    while self.middle_consonant_letters_count[middle_consonant_letter_index] < self.max_consonant_letter_count:
                        first_consonant_letter_index = random.randrange(0,self.first_consonant_letter_number)
                        last_consonant_letter_index = random.randrange(0,self.last_consonant_letter_number)
                        if self.first_consonant_letters_count[first_consonant_letter_index] < max_first_consonant_letter_count and self.last_consonant_letters_count[last_consonant_letter_index] < max_last_consonant_letter_count:
                            self.pick_letter_list.append(self.get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                            # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                            self.first_consonant_letters_count[first_consonant_letter_index] += 1
                            self.middle_consonant_letters_count[middle_consonant_letter_index] += 1
                            self.last_consonant_letters_count[last_consonant_letter_index] += 1

            elif self.criteria_consonant_letter == "first":
                max_middle_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.first_consonant_letter_number / self.middle_consonant_letter_number)
                max_last_consonant_letter_count = math.ceil(self.max_consonant_letter_count * self.first_consonant_letter_number / self.last_consonant_letter_number)
                for first_consonant_letter_index in range(self.first_consonant_letter_number):
                    while self.first_consonant_letters_count[first_consonant_letter_index] < self.max_consonant_letter_count:
                        middle_consonant_letter_index =  random.randrange(0,self.middle_consonant_letter_number)
                        last_consonant_letter_index = random.randrange(0,self.last_consonant_letter_number)
                        if self.middle_consonant_letters_count[middle_consonant_letter_index] < max_middle_consonant_letter_count and self.last_consonant_letters_count[last_consonant_letter_index] < max_last_consonant_letter_count:
                            self.pick_letter_list.append(self.get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                            # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                            self.first_consonant_letters_count[first_consonant_letter_index] += 1
                            self.middle_consonant_letters_count[middle_consonant_letter_index] += 1
                            self.last_consonant_letters_count[last_consonant_letter_index] += 1

            print(self.pick_letter_list)
            print(len(self.pick_letter_list))
            print(sum(self.first_consonant_letters_count), sum(self.middle_consonant_letters_count),sum(self.last_consonant_letters_count))
            print(self.first_consonant_letters_count)
            print(self.middle_consonant_letters_count)
            print(self.last_consonant_letters_count)
            if self.check_data_is_representative(1):
                break
            else:
                self.reset_variable()

        return self.pick_letter_list

    def cp_sampling_file(self,from_path = "H:/data/Hangul_Characters_Image128/", to_path = "H:/data/Hangul_Characters_Image128_radomSampling"):
        os.makedirs(to_path,exist_ok=True)
        for letter in self.pick_letter_list:
            # os.makedirs(os.path.join(to_path,letter), exist_ok=True)
            shutil.copytree(os.path.join(from_path,letter),os.path.join(to_path,letter))

if __name__ == "__main__":
    data = make_sampling_data(max_consonant_letter_count=20,criteria_consonant_letter="middle")
    sampling_letter = data.letter_random_sampling()
    print(sampling_letter)
    data.cp_sampling_file()