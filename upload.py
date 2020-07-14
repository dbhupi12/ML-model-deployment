from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, AUDIO

import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
import pickle
import tensorflow as tf
from keras import models
from keras import layers
import warnings
#import scipy.io
warnings.filterwarnings('ignore')

data = pd.read_csv('dataset(dry n wet cough).csv')

#Encoding the Labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

app = Flask(__name__)

audio = UploadSet('audio', AUDIO)

app.config['UPLOADED_AUDIO_DEST'] = 'static/audio'
configure_uploads(app, audio)

new_model = tf.keras.models.load_model('(dry n wet cough)ANN_model')
#scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('upload.html') 
 
@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'audio' in request.files:
        filename = audio.save(request.files['audio'])
        #print(filename)
        #sr, y = scipy.io.wavfile.read(f'static/audio/{filename}')
        y, sr = librosa.load(f'static/audio/{filename}', mono=True)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        a = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]    
        for e in mfcc:
            a.append(np.mean(e))

        a = np.array(a).reshape(1,26)
        check = scaler.transform(a)
        y_pred = new_model.predict(check)
        if y_pred > 0.5 :
            output = 1
        else :
            output = 0

        return render_template('upload.html', prediction_text='Type of cough is {}'.format(y_pred))


if __name__ == "__main__":
    app.run(debug=True)