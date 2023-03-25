from jamo import h2j,j2hcj
import torch.nn as nn
import re

Chosung_list = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

Jungsung_list = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

Jongsung_list = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

Base_code, Chosung, Jungsung = 44032, 558, 28

def separation_1(keyword: str) -> list(int):
    '''
    단어가 들어오면 초성, 중성, 종성으로 나눠서 embedding 해주는 함수
    '''
    result = [0]*68

    #초성
    char_code = ord(keyword) - Base_code
    char1 = int(char_code/Chosung)
    result[char1] = 1

    #중성
    char2 = int((char_code-(Chosung*char1))/Jungsung)
    result[char2] = 1

    #종성
    char3 = int((char_code-(Chosung*char1)-(Jungsung*char2)))
    result[char3] = 1

    print("".join(result))
    return result



