from natasha import(
    NamesExtractor,
    PersonExtractor,
    LocationExtractor,
    AddressExtractor,
    OrganisationExtractor,
    DatesExtractor,
    MoneyExtractor)
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import wikipediaapi 
import pymorphy2 
from nltk.corpus import wordnet 
from translate import Translator 
from nltk.corpus.reader.wordnet import WordNetError 
import re 
import time 
start = time.time()  


def preprocess_text(text): 
    stop_words = set(stopwords.words('russian')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    text = ' '.join(filtered_sentence) 
    return text 


def word_meaning(word): 
    morph = pymorphy2.MorphAnalyzer()
    pr = morph.parse(word)[0] 
    word = pr.normal_form 
    wiki = wikipediaapi.Wikipedia('ru') 
    page_py = wiki.page(word) 

    name = page_py.summary.split(".\n")[0] 
    regex = re.compile(".*?\((.*?)\)") 
    result = re.findall(regex, name) 

    name = name.replace("(", "") 
    name = name.replace(")", "")
    try:
        name = name.replace(result[0], "")
    except IndexError:
        pass
    name = name.split(" ") 

    words = []  
    for i in name: 
        p = morph.parse(i)[0]
        if p.tag.POS == 'NOUN':
            words.append(p.normal_form)
    array_val = dict() # словарь для значений схожести сущиствительных с настоящим словом

    for i in words:
        translator = Translator(to_lang="en",from_lang="ru")
        trans_word = translator.translate(i)
        word_en = translator.translate(word)
        if str(trans_word).__contains__("the"):
            trans_word = str(trans_word).replace("the ", "")
        if str(word_en).__contains__("the"):
            word_en = str(word_en).replace("the ", "")
        if str(trans_word).__contains__("a "):
            trans_word = str(trans_word).replace("a ", "")
        if str(word_en).__contains__("a "):
            word_en = str(word_en).replace("a ", "")
        if str(trans_word).__contains__("an "):
            trans_word = str(trans_word).replace("an ", "")
        if str(word_en).__contains__("an "):
            word_en = str(word_en).replace("an ", "")
        try:
            w1 = wordnet.synset(trans_word + ".n.01")
        except WordNetError:
            continue
        try:
            w2 = wordnet.synset(word_en + ".n.01")
        except WordNetError:
            continue
        if i != word:
            array_val[i] = w1.wup_similarity(w2)
    another = dict()
    array_val = sorted(array_val.items(), key=lambda kv: (kv[1], kv[0]))
    for i in array_val: 
        if str(i[1]) != "1.0":
            another[i] = i[1]
    array_val = dict(array_val)
    another = sorted(array_val.items(), key=lambda kv: (kv[1], kv[0])) 
    another.reverse()
    another_rer = []
    for i in another:
        if i[1] == 1:
            continue
        another_rer.append(i)
    return another_rer[0][0], another_rer[0][1]


def main():
    extractor = [NamesExtractor(), PersonExtractor(), LocationExtractor(),
                 AddressExtractor(), OrganisationExtractor(), DatesExtractor(), MoneyExtractor()]
    potolok = ['Имя', 'Человек', 'Место', 'Адрес', 'Организация', 'Время', 'Деньги']
    text = 'Президентом России является Владимир Владимирович Путин, штаб-квартира которого находится' \
           'в г. Москва организации "ТОО Кремль" и если в полдень пройтись по площади Кремля можно увидеть как люди' \
           'толпами тратят рубли на реконструкцию площади ' # наглядный пример
    word_matches = []
    for i in range(len(extractor)):
        matches = extractor[i](text)
        try:
            for match in matches:
                start, stop = match.span
                word_matches.append(text[start:stop])
                print(text[start:stop], end="\t-\t"), print(potolok[i])
        except IndexError:
            pass
    full_text = text
    for i in word_matches:
        full_text = full_text.replace(i, "")
    full_text = full_text.replace("-", " ")
    full_text = re.sub(r"#\S+", "", full_text)
    full_text = re.sub(r"@\S+", "", full_text)
    full_text = re.sub(r"pic\S+", "", full_text)
    full_text = re.sub(r"http\S+", "", full_text)
    full_text = re.sub(r"https\S+", "", full_text)
    full_text = re.sub(r"RT\S+", "", full_text)
    full_text = re.sub("\d+", "", full_text)
    full_text = re.sub(r'[^\w\s]', '', full_text)
    full_text = full_text.split(" ")
    got_text = []
    for i in full_text:
        if i != "":
            got_text.append(i)

    for i in got_text:
        try:
            a,b = word_meaning(i)
        except IndexError:
            continue
        print(i,end="\t-\t")
        print(str(a)+" |с точностью в "+str(b))


main()
print(time.time()-start)
