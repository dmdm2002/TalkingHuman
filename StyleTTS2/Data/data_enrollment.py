
f = open("D:/Side/SieProject/StyleTTS/dataset/transcript.v.1.4.txt", 'r', encoding='utf-8')
new_f = open('D:/Side/SieProject/StyleTTS/dataset/kss_new.txt', 'w')
for line in f:
    print(line)
    info_list = line.split('|')[:2]
    new_line = f'{info_list[0]}|{info_list[1]}|0'
    new_f.write(f'{new_line}\n')
f.close()
new_f.close()