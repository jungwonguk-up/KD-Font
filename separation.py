from jamo import h2j,j2hcj
import torch.nn as nn
import re

Chosung_list = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

Jungsung_list = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

Jongsung_list = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

Base_code, Chosung, Jungsung = 44032, 558, 28

def separation_1(keyword: str) -> list:
    '''
    단어가 들어오면 초성, 중성, 종성으로 나눠서 embedding 해주는 함수
    '''
    assert len(keyword) == 1 # 글자 길이 검사
    assert ord('가') <= ord(keyword) <= ord('힣') # keyword 가 한글 완성 문자인지 검사 44032 ~ 55203

    result = [0] * 68
    str_result = ['0'] * 68

    #초성
    char_code = ord(keyword) - Base_code
    char1 = int(char_code/Chosung)
    result[char1] = 1
    str_result[char1] = '1'

    #중성
    char2 = int((char_code-(Chosung*char1))/Jungsung)
    result[char2] = 1
    str_result[char2] = '1'

    #종성
    char3 = int((char_code-(Chosung*char1)-(Jungsung*char2)))
    result[char3] = 1
    str_result[char3] = '1'

    print("".join(str_result))
    return result


if __name__ == "__main__":
    input_string = input("test input: ")
    # separation_1(input_string)
    print(separation_1(input_string))
