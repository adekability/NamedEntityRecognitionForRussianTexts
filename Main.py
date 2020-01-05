from natasha import( # библиотека обработки русского текста
    NamesExtractor,
    PersonExtractor,
    LocationExtractor,
    AddressExtractor,
    OrganisationExtractor,
    DatesExtractor,
    MoneyExtractor)
from nltk.corpus import stopwords # метод библиотеки nltk для удаления стоп-слов
from nltk.tokenize import word_tokenize # метод библиотеки nltk для токенайзинга
import wikipediaapi # API сайта wikipedia.org
import pymorphy2 # библиотека обработки русских слов
from nltk.corpus import wordnet # библиотека сравнения схожести слов с применением машинного обучения
from translate import Translator # библиотека переводчик с русского на английский
from nltk.corpus.reader.wordnet import WordNetError # импорт ошибки WordNetError для поимки в try, except
import re # импорт библиотеки регулярных выражений
import time # импорт библиотеки времмени
start = time.time()  # время начала программы в переменную


def preprocess_text(text): # метод по удалению стоп-слов
    stop_words = set(stopwords.words('russian')) # передаем методу язык "русский"
    word_tokens = word_tokenize(text) # выполняем токенайзинг
    filtered_sentence = [w for w in word_tokens if not w in stop_words] # удаляем стоп слова и оставляем в список
    text = ' '.join(filtered_sentence) # список в строку
    return text # возвращаем значение строки


def word_meaning(word): # основной метод обработки
    morph = pymorphy2.MorphAnalyzer() # вызываем метод для перевода в нормальную форму слова
    pr = morph.parse(word)[0] # # перевод в метод обработки нужного слова
    word = pr.normal_form # перевод слова в нормальную форму
    wiki = wikipediaapi.Wikipedia('ru') # вызов русской википедии
    page_py = wiki.page(word) # поиск слова в русской википедии

    name = page_py.summary.split(".\n")[0] # берем полный текст со страницы, и отделяем лишь первое предложение
    regex = re.compile(".*?\((.*?)\)") # находим все слова в скобках (в переводах, обозначениях)
    result = re.findall(regex, name) # переменная со словами в скобках

    name = name.replace("(", "") # удаляем все скобки
    name = name.replace(")", "")# удаляем все скобки
    try:
        name = name.replace(result[0], "") # удаляем все слова в скобке, находится в try, так как скобок может и не быть
    except IndexError:
        pass
    name = name.split(" ") # переводим все слова в список

    words = []  # пустая переменная для поиска существительных по слову
    for i in name: # цикл для поиска сущиствительных
        p = morph.parse(i)[0]
        if p.tag.POS == 'NOUN':
            words.append(p.normal_form)
    array_val = dict() # словарь для значений схожести сущиствительных с настоящим словом

    for i in words:# цикл для перевода слов в английский, и сопоставления с настоящим словом по схожести, после чего берем коэффициент схожести
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
            w1 = wordnet.synset(trans_word + ".n.01")# здесь задаем итерационное слово, обучаем в методе итерируемое слово
        except WordNetError:
            continue
        try:
            w2 = wordnet.synset(word_en + ".n.01")# здесь задаем настоящее слово
        except WordNetError:
            continue
        if i != word:
            array_val[i] = w1.wup_similarity(w2)# сверяем по схожести, сохраняем в список результат, проводится обучение,
            # и сравнивание схожести с настоящим словом, далее результат каждого слова зависит от предыдущего
    another = dict()# словарь для массива и сортировки
    array_val = sorted(array_val.items(), key=lambda kv: (kv[1], kv[0])) # сортируем по значениям
    for i in array_val: # удаляем схожие слова с настоящим
        if str(i[1]) != "1.0":
            another[i] = i[1]
    array_val = dict(array_val) # обратно переводим в словарь
    another = sorted(array_val.items(), key=lambda kv: (kv[1], kv[0])) # сортируем по значениям
    another.reverse()# переворачиваем массив
    another_rer = [] # новый массив для сохранения результата наиболее похожего слова
    for i in another:
        if i[1] == 1:
            continue
        another_rer.append(i)
    return another_rer[0][0], another_rer[0][1] # возвращаем сущность


def main():# main метод
    extractor = [NamesExtractor(), PersonExtractor(), LocationExtractor(),
                 AddressExtractor(), OrganisationExtractor(), DatesExtractor(), MoneyExtractor()] # массив с методами извлечения имен, личностей, мест и т.д.
    potolok = ['Имя', 'Человек', 'Место', 'Адрес', 'Организация', 'Время', 'Деньги']
    text = 'Президентом России является Владимир Владимирович Путин, штаб-квартира которого находится' \
           'в г. Москва организации "ТОО Кремль" и если в полдень пройтись по площади Кремля можно увидеть как люди' \
           'толпами тратят рубли на реконструкцию площади ' # наглядный пример
    word_matches = []
    for i in range(len(extractor)):# цикл, в котором библиотека natasha будет извлекать сущности из массива методов extractor
        matches = extractor[i](text)
        try:
            for match in matches:
                start, stop = match.span
                word_matches.append(text[start:stop])
                print(text[start:stop], end="\t-\t"), print(potolok[i])
        except IndexError:
            pass
    #здесь производится удаление ненужных символов
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

    # в этом цикле вызывается метод word_meaning, возвращаемый слово и его сущность
    for i in got_text:
        try:
            a,b = word_meaning(i)
        except IndexError:
            continue
        print(i,end="\t-\t")
        print(str(a)+" |с точностью в "+str(b))


main()
print(time.time()-start) # конец программы и потраченное время