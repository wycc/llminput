import itertools
import re

def load_cin_file(typ, filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    key_to_character = {}
    if type == 'raw':
        is_def = True
    else:
      is_def = False
    endkey=r'[ ]' # This is default endkey, which is used when %endkey is not found in the file
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # 忽略空行和注释行
            continue
        # handle %chardef to load key sequence only in the chardef section
        if typ == 'cin':
          if line.startswith("%chardef"):
              tokens = line.split()
              if tokens[1] == 'begin':
                  is_def = True
              elif tokens[1] == 'end':
                  is_def = False
              continue
          elif line.startswith("%endkey"):
              tokens = line.split()
              endkey = '['+tokens[1]+' ]'
              continue
          if is_def == False:
              continue
        # split the line by any characters in endkey
        parts = line.split(' ')

        if len(parts) >= 2:
            if typ == 'raw':
              key_sequence = parts[0].upper()
              character = parts[1]
            else:
              key_sequence = parts[1].upper()
              character = parts[0]
                
            if key_sequence in key_to_character:
                key_to_character[key_sequence].append(character)
            else:
                key_to_character[key_sequence] = [character]
    
    return key_to_character

def convert_key_sequence_to_text(key_sequence, key_to_character):
    key_sequence = key_sequence.upper()
    #split the key sequence by space
    key_sequence = key_sequence.split()
    # Each key sequence might map to multiple words, we need to generate all posible combinations
    # of words that can be generated from the key sequence
    # We can use itertools.product to generate all possible combinations of words
    # from the key sequence. We can then join the words together to form the final text.
    words = [key_to_character[key] for key in key_sequence]
    all_combinations = itertools.product(*words)
    all=[]
    for combination in all_combinations:
        all.append(''.join(combination))
    return all

# 示例Cin格式文件路径
#cin_file_path = '14_dayi.cin'
cin_file_path = 'lookup-bpmf.cin'

# 载入Cin格式输入法表格
#key_to_character = load_cin_file('raw',cin_file_path)
key_to_character = load_cin_file('cin',cin_file_path)

# 示例按键序列
key_sequence = 'u ek7 bp6'
# 转换按键序列为文字
text = convert_key_sequence_to_text(key_sequence, key_to_character)
print(f"转换后的文字: {text}")
