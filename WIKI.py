from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

###################################################
# Görev 1: Metin Ön İşleme İşlemlerini Gerçekleştiriniz
###################################################

df = pd.read_csv("datasets/wiki_data.csv", sep=",")
df.head()

# Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz.

def clean_text(dataframe):
    dataframe['text'] = dataframe['text'].str.lower()
    dataframe['text'] = dataframe['text'].str.replace('[^\w\s]', '')
    dataframe['text'] = dataframe['text'].str.replace('\d', '')
    return dataframe

# Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df = clean_text(df)

# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri (ben, sen, de, da, ki, ile vs) çıkaracak remove_stopwords adında
# fonksiyon yazınız.

import nltk

def remove_stopwords(dataframe):
    sw = stopwords.words('english')
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return dataframe

# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df = remove_stopwords(df)

# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.

def rare_words(dataframe,th=100):
    temp_df = pd.Series(' '.join(dataframe['text']).split()).value_counts()
    drops = temp_df[temp_df <= th]
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return dataframe

df = rare_words(df,100)

# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.

def tokenization(dataframe):
    dataframe["text"] = dataframe["text"].apply(lambda x: TextBlob(x).words)
    return dataframe

df = tokenization(df)

# Adım 7: Lemmatization işlemi yapınız.

def lemmatization(dataframe):
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x]))
    return dataframe

df = lemmatization(df)

df.head()
###################################################
# Görev 2: Veriyi Görselleştiriniz
###################################################

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.

tf[tf["tf"] > 6000].plot.bar(x="words", y="tf")
plt.show()

# Adım 3: Kelimeleri WordCloud ile görselleştiriniz.

text = " ".join(i for i in df.text)


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

###################################################
#Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
###################################################

# Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.
# Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# Adım 3: Fonksiyonu açıklayan 'docstring' yazınız.

def wiki_prep(dataframe,col,rare_th=100,barplot=False,wordcloud=False):
    """
    Kullanıcı tarafından verilecek kolon ve dataframe bilgisine göre text tipinde olan kolonlara metin ön işleme adımlarını uygular
    ve bu metni görselleştirir.
    Parameters
    -----
        dataframe: pandas dataframe
                Üzerinde ön işleme işlemi yapılacak dataframe
        col: string
                Üzerinde işlem yapılacak kolon
        rare_th:int
                Metinde nadir bulunan ve önemsiz olduğu düşünülen kelimelerin frekansının eşik değeri
        barplot:bool
                Kelimelerin frekansları barplot ile görselleştirilir
        wordcloud:bool
                Kelimelerin frekansları wordcloud ile görselleştirilir.
    Returns
    -----
        dataframe:pandas dataframe
                Üzerinde metin ön işleme yapılan dataframe
    Examples
    -----
        df = pd.read_csv("datasets/wiki_data.csv", sep=",")
        df = wiki_prep(df,"text",100,barplot=True,wordcloud=True)

    """
    dataframe = clean_text(dataframe)
    dataframe = remove_stopwords(dataframe)
    dataframe = rare_words(dataframe,100)
    dataframe = tokenization(dataframe)
    dataframe = lemmatization(dataframe)
    if barplot:
        tf = dataframe[col].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf.sort_values("tf", ascending=False)
        tf[tf["tf"] > 6000].plot.bar(x="words", y="tf")
        plt.show()
    if wordcloud:
        text = " ".join(i for i in dataframe[col])
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    return dataframe

df = wiki_prep(df,"text",100,barplot=True,wordcloud=True)
df.head()
















