import random,math


def get_letter_by_unicode(index_list):
    return chr((index_list[0] * 588) + (index_list[1] * 28) + (index_list[2]) + 44032)
def letter_random_sampling(max_consonant_letter_count = 10, criteria_consonant_letter = "last"):
    # 초성, 중성 종성의 개수
    first_consonant_letter_number = 19
    middle_consonant_letter_number = 21
    last_consonant_letter_number = 28

    # 초성, 중성, 종성 각각의 문자가 사용된 개수 리스트
    first_consonant_letters_count = [0 for _ in range(first_consonant_letter_number)] #ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
    middle_consonant_letters_count = [0 for _ in range(middle_consonant_letter_number)] #ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ
    last_consonant_letters_count = [0 for _ in range(last_consonant_letter_number)] # blank's index is zero # 'blank' + ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ

    pick_letter_list = []
    # pick_unicode_list = []

    if criteria_consonant_letter == "last":
        max_first_consonant_letter_count = math.ceil(max_consonant_letter_count * last_consonant_letter_number / first_consonant_letter_number )
        max_middle_consonant_letter_count = math.ceil(max_consonant_letter_count * last_consonant_letter_number / middle_consonant_letter_number)
        for last_consonant_letter_index in range(last_consonant_letter_number):
            while last_consonant_letters_count[last_consonant_letter_index] < max_consonant_letter_count:
                first_consonant_letter_index =  random.randrange(0,first_consonant_letter_number)
                middle_consonant_letter_index =  random.randrange(0,middle_consonant_letter_number)
                if first_consonant_letters_count[first_consonant_letter_index] < max_first_consonant_letter_count and middle_consonant_letters_count[middle_consonant_letter_index] < max_middle_consonant_letter_count:
                    pick_letter_list.append(get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                    # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                    first_consonant_letters_count[first_consonant_letter_index] += 1
                    middle_consonant_letters_count[middle_consonant_letter_index] += 1
                    last_consonant_letters_count[last_consonant_letter_index] += 1

    elif criteria_consonant_letter == "middle":
        max_first_consonant_letter_count = math.ceil(max_consonant_letter_count * middle_consonant_letter_number / first_consonant_letter_number )
        max_last_consonant_letter_count = math.ceil(max_consonant_letter_count * middle_consonant_letter_number / last_consonant_letter_number)
        for middle_consonant_letter_index in range(middle_consonant_letter_number):
            while middle_consonant_letters_count[middle_consonant_letter_index] < max_consonant_letter_count:
                first_consonant_letter_index = random.randrange(0,first_consonant_letter_number)
                last_consonant_letter_index = random.randrange(0,last_consonant_letter_number)
                if first_consonant_letters_count[first_consonant_letter_index] < max_first_consonant_letter_count and last_consonant_letters_count[last_consonant_letter_index] < max_last_consonant_letter_count:
                    pick_letter_list.append(get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                    # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                    first_consonant_letters_count[first_consonant_letter_index] += 1
                    middle_consonant_letters_count[middle_consonant_letter_index] += 1
                    last_consonant_letters_count[last_consonant_letter_index] += 1

    elif criteria_consonant_letter == "first":
        max_middle_consonant_letter_count = math.ceil(max_consonant_letter_count * first_consonant_letter_number / middle_consonant_letter_number)
        max_last_consonant_letter_count = math.ceil(max_consonant_letter_count * first_consonant_letter_number / last_consonant_letter_number)
        for first_consonant_letter_index in range(first_consonant_letter_number):
            while first_consonant_letters_count[first_consonant_letter_index] < max_consonant_letter_count:
                middle_consonant_letter_index =  random.randrange(0,middle_consonant_letter_number)
                last_consonant_letter_index = random.randrange(0,last_consonant_letter_number)
                if middle_consonant_letters_count[middle_consonant_letter_index] < max_middle_consonant_letter_count and last_consonant_letters_count[last_consonant_letter_index] < max_last_consonant_letter_count:
                    pick_letter_list.append(get_letter_by_unicode([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index]))
                    # pick_unicode_list.append([first_consonant_letter_index, middle_consonant_letter_index, last_consonant_letter_index])
                    first_consonant_letters_count[first_consonant_letter_index] += 1
                    middle_consonant_letters_count[middle_consonant_letter_index] += 1
                    last_consonant_letters_count[last_consonant_letter_index] += 1

    print(pick_letter_list)
    print(len(pick_letter_list))
    print(sum(first_consonant_letters_count), sum(middle_consonant_letters_count), sum(last_consonant_letters_count))
    print(first_consonant_letters_count)
    print(middle_consonant_letters_count)
    print(last_consonant_letters_count)
    # print(max_middle_consonant_letter_count, max_fir_consonant_letter_count)

