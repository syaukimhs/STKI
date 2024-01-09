import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud

# menampilkan judul halaman
st.title("Sistem Analis Sentimen Ulasan Aplikasi E-commerce di Playstore")

# membuat deskripsi dari sistem analis
st.write("""
         Sistem analis sentimen ulasan aplikasi e-commerce merupakan sebuah sistem 
         untuk mengetahui jenis ulasan positif atau negatif dari user, tujuan dari pembuatan
         sistem ini yaitu untuk mempermudah developer guna mengetahui kekurangan dari aplikasinya
         ini dibuat dengan menggunakan menggunakan bahasa pemrograman python. 
         Untuk modelling yang dipakai dalam pembuatan aplikasi ini yaitu 
         menggunakan model naive bayes.
    """)

# memanggil dataset
st.subheader('Menampilkan 5 dataset pertama')
dataset = pd.read_csv('data_clean.csv')
data = pd.DataFrame(dataset)
st.write(data.head())

# menampilkan visualisasi perbandingan label sebelum di smote
plt.figure(figsize=(5, 3))
sns.countplot(data=data, x='Label', palette={0: "red", 1: "skyblue"})
plt.title('Visualisasi Sentimen Positif dan Negatif')
plt.xlabel('Label')
plt.ylabel('Jumlah')
st.pyplot(plt)

# proses feature enginering
x = data['clean_teks']
y = data['Label']

vec_TF_IDF = TfidfVectorizer(ngram_range=(1,1))
vec_TF_IDF.fit(x)

x1 = vec_TF_IDF.transform(x).toarray()
data_tabular_tf_idf = pd.DataFrame(x1,columns=vec_TF_IDF.get_feature_names_out())

x_train = np.array(data_tabular_tf_idf)
y_train = np.array(y)

chi2_features = SelectKBest(chi2, k=3000)
x_kbest_features = chi2_features.fit_transform(x_train, y_train)

selected_x = x_kbest_features
selected_x

x = selected_x
y = data.Label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

smote = SMOTE(random_state=0)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
sentimen_counts = y_train_resampled.value_counts()

# menampilkan visualisasi perbandingan label sesudah di smote
plt.figure(figsize=(5, 3))
plt.bar(sentimen_counts.index, sentimen_counts.values, color=['red', 'skyblue'])
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.title('Visualisasi data dengan menggunakan SMOTE')
plt.xticks(sentimen_counts.index, ['0', '1'])
st.pyplot(plt)


# visualisasi terhadap data negatif
data_negatif = data[data['Label'] == 0]
data_positif = data[data['Label'] == 1]

all_text_s0 = ' '.join(word for word in data_negatif["ulasan"])
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(9, 6))
plt.imshow(wordcloud.to_image(), interpolation='bilinear')
#plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Visualisasi Data Negatif')
plt.margins(x=0, y=0)
st.pyplot(plt)

# visualisasi terhadap data positif
all_text_s1 = ' '.join(word for word in data_positif["ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
#wordcloud = WorldCloud(colormap='Blues', width=1000, mode="RGBA", background_color='white').generate(all_text_s1)
plt.figure(figsize=(9,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Visualisasi Data Positif")
plt.margins(x=0, y=0)
st.pyplot(plt)


# melakukan proses load model
model_fraud = pickle.load(open('model_fraud.sav','rb'))

tfidf = TfidfVectorizer
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf_idf.sav","rb"))))

# melakukan proses input ulasan
clean_teks = st.text_input('Input Kalimat')

fraud_detection = ''

# membuat button dan fungsinya
if st.button('hasil analisis'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))
    
    if(predict_fraud == 1):
        fraud_detection == 'Ulasan Positif'
    else:
        fraud_detection == 'Ulasan Negatif'

st.success(fraud_detection)