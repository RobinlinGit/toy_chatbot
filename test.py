from utils import load_lines


filename = 'C://NLP//clean_chat_corpus//chatterbot.tsv'
lines = load_lines(filename)
for line in lines[:10]:
    print(line)
