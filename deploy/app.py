import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask, redirect, url_for, request, render_template
import tensorflow_hub as hub
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa
from pathlib import Path
import os, shutil
import random
from tensorflow.keras.preprocessing import image as kimage
plt.ioff()
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pandas as pd
import glob

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# define a Flask app
app = Flask(__name__)

######### MONODIM ##################################################

# MONDOIM MODEL DEFINITION

base_net = resnet50.ResNet50(weights="imagenet", include_top=False,
	input_shape=(224, 224, 3), pooling="avg")
for layer in base_net.layers:
    layer.trainable = False
# Output of the base_net model
x = base_net.output
# intermediate fully-connected layer + ReLU
x = keras.layers.Dense(512, activation='relu')(x)
# final fully-connected layer + SoftMax 
pred = keras.layers.Dense(10, activation='softmax')(x)

MODELAUDIO = keras.Model(inputs=base_net.input, outputs=pred)
MODELAUDIO.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
MODELAUDIO.load_weights('monodim_weights')


print('Successfully loaded model...')

# PREPROCESSING FUNCTION

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform(file_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, audio_binary


frame_length = 2048
frame_step = 512
num_mel_bins = 75
num_spectrogram_bins = (frame_length // 2) + 1
fmin = 0.0
sample_rate = 44100
fmax = sample_rate / 2


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    #zero_padding = tf.zeros([140000] - tf.shape(waveform), dtype=tf.float32) # NON SUPERARE I 3 SECONDI CON TIME STRETCH
    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = waveform
    magnitude_spectrograms  = tf.signal.stft(
      equal_length, frame_length, frame_step)
    magnitude_spectrograms  = tf.abs(magnitude_spectrograms)
    
    # Step: magnitude_spectrograms->mel_spectrograms
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]


    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, fmin,
        fmax)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
  linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms

def draw_spectrogram(spectrogram, output_dir_path, i):
    fig, ax = plt.subplots(figsize=(20,20))
    mfcc_data= np.swapaxes(spectrogram, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.axis("off")
    fig.savefig(f'{output_dir_path}/to_predict/mel_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# PREDICT FUNCTION

def model_predict_audio(img_path, model):
    audio_p = "upload"

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        audio_p,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        shuffle=True,
        batch_size=1)

    test, label = next(iter(test_generator))
    tf_model_predictions = model.predict(test)
    tf_pred_dataframe = pd.DataFrame(tf_model_predictions)
    tf_pred_dataframe.columns = ['moose','buffalo', 'deer', 'horse' ,'otter' ,'sheep', 'chimpanzee', 'lion', 'raccoon' ,'fox'] #Qui specifichi l'etichetta che ti printa dopo il modello
    print(len(tf_pred_dataframe.columns))
    predicted_ids = np.argmax(tf_model_predictions, axis=-1)
    predicted_labels = tf_pred_dataframe.columns[predicted_ids]
    
    return predicted_labels

# DEFINE ROUTES

@app.route('/predict_audio', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']  
        
        directory = "upload/to_predict"

        #delete other files if present
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))    
        
        # save the file to ./uploads
        basepath = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join("upload/to_predict", f.filename) 

         

        mel_path = "upload/to_predict/mel_1.png"
        print(filepath)
        f.save(filepath)

        waveform1, audio_binary1 = get_waveform(filepath)
        mel = get_spectrogram(waveform1)
        draw_spectrogram(mel, "upload", 1)
        try:
            os.remove(filepath)
        except OSError:
            pass
        
        # make prediction about this image's class
        preds = model_predict_audio(mel_path, MODELAUDIO)[0]
        
        result = str(preds)
        print('[PREDICTED CLASS]: {}'.format(preds))

        #delete file uploaded
        try:
            os.remove(filepath)
            os.remove(mel_path)
        except OSError:
            pass

        return result
    
    return None


######### BIDIM ##################################################

#BIDIM MODEL DEFINITION

base_net_bdim = resnet50.ResNet50(weights="imagenet", include_top=False,
	input_shape=(224, 224, 3), pooling="avg")
for layer in base_net_bdim.layers:
    layer.trainable = False
# Output of the base_net model
x = base_net_bdim.output
# intermediate fully-connected layer + ReLU
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# final fully-connected layer + SoftMax 
pred = tf.keras.layers.Dense(10, activation='softmax')(x)
MODELIMAGE = tf.keras.Model(inputs=base_net_bdim.input, outputs=pred)
MODELIMAGE.load_weights('bidim_weights')


print('Successfully loaded model...')

#PREDICTION FUNCTION

def model_predict_image(img_path, model):
    '''
        helper method to process an uploaded image
    '''
    img_p = "upload"
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        img_p,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        shuffle=True,
        batch_size=1)

    test, label = next(iter(test_generator))
    print(len(test))
    tf_model_predictions = model.predict(test)
    tf_pred_dataframe = pd.DataFrame(tf_model_predictions)
    tf_pred_dataframe.columns = ["buffalo", "moose", "deer", "horse", "otter", "sheep", "chimpanzee", "lion", "raccoon", "fox"]  #Qui specifichi l'etichetta che ti printa dopo il modello
    predicted_ids = np.argmax(tf_model_predictions, axis=-1)
    predicted_labels = tf_pred_dataframe.columns[predicted_ids]
    print(predicted_labels)
    
    return predicted_labels

#PREDICT ROUTE

@app.route('/predict_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        directory = "upload/to_predict"

        #delete other files if present
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))    

        # save the file to ./uploads
        basepath = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join("upload/to_predict", f.filename)
                 
        f.save(filepath)

        # make prediction about this image's class
        preds = model_predict_image(filepath, MODELIMAGE)[0]
        
        result = str(preds)
        print('[PREDICTED CLASS]: {}'.format(preds))

        #delete file uploaded
        try:
            os.remove(filepath)
        except OSError:
            pass
        
        return result
    
    return None



######### RETRIEVAL ################################################



my_resnet50 = resnet50.ResNet50(include_top = False, weights='imagenet',
                       pooling = 'max', input_shape=(224,224,3))

def resnet50_features(img, net):
    '''
      Takes an image in order to extract the features of resnet50
      @params:
        - img: image to compute
        - net: neural network to use
    '''
    x = kimage.img_to_array(img) # to numpy
    x = resnet50.preprocess_input(x) # preprocess for network
    x = np.expand_dims(x, axis=0) # necessario per la rete
    features = net.predict(x).flatten()
    return features

# Limit number of loaded images
maximg_class = 200 # 200 img per ogni classe

classes = ["buffalo", "moose", "deer", "horse", "otter", "sheep", "chimpanzee",
           "lion", "raccoon", "fox"]

# Data loader
def load_data(base_path, net, feature_extractor=resnet50_features):
    '''
      Load image database features by applying feature extraction of a neural network
      @params:
        - base_path: path where folders of classes of images are stored
        - feature_extractor: function that extracts features on a image
        - preprocess_fuction: function to apply to image in order to insert it 
              in the neural network
        - net: neural network to apply
    '''
    paths = []
    features = []

    for fold in classes:
      cur_fold = base_path + fold + '/'
      for file_n, f in enumerate(sorted(os.listdir(cur_fold))):
        if f.endswith('.jpg'):
          # Save file path
          cur_path = cur_fold + f
          paths.append(cur_path)
          
        if (file_n > maximg_class) :
          break
          
      print(f"{fold} DONE")

    features = np.array(features)
    return features, paths


jpg_path = f"retrieval/images_animals10_small/" # PATH DELLE IMMAGINI
_, paths = load_data(jpg_path, feature_extractor=resnet50_features, net = my_resnet50)

# Load saved features PATH DOVE HO SALVATO LO SPAZIO DI FEATURES
X_train = np.load("retrieval/features_resnet50.npy")

from sklearn.neighbors import KDTree
tree = KDTree(X_train)


@app.route('/retrieval', methods=['GET', 'POST'])
def upload_image_retrieval():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        directory = "upload/to_predict"

        #delete other files if present
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))    

        retrdir= "static/results"
        
        try:
            os.makedirs(f"static/results")
        except OSError:
            pass


        for filename in os.listdir(retrdir):
            file_path = os.path.join(retrdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))    


        print("pre-filepath")
        # save the file to ./uploads
        basepath = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join("upload/to_predict", f.filename)
                 
        f.save(filepath)
        print("saved")
        # make prediction about this image's class
        
        # carico immagine di query + estraggo le sue features
        # PATH PER QUERY IMAGE
        query_image = kimage.load_img(filepath, target_size=(224, 224, 3))
        print("saved")
        query_features = resnet50_features(query_image, my_resnet50)
        print("saved")
        query_features = np.expand_dims(query_features, axis = 0)
        print("saved")

        # ricerca nello spazio 
        dist, ind = tree.query(query_features, k=10)
        print("saved")

        try:
            os.makedirs(f"static/results")
        except OSError:
            pass

        # print best 10
        for j in range(2):
            for i in range(5):
                path = paths[ind[0][i+j*5]]
                print(path)
                classe = path.split('/')[-2]
                im = kimage.load_img(path, target_size=(224,224,3))
                plt.imshow(im)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.savefig(f"static/results/out_{i+j*5}.png", bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                print("saved")

        print("exit loop")
        #delete file uploaded
        try:
            os.remove(filepath)
        except OSError:
            pass
        print("almost return")
        return ""
    
    return None



###################################################################


#RENDER TEMAPLATES ROUTES

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/audio/', methods=['GET'])
def audio():
    return render_template("monodim_app.html")

@app.route('/image/', methods=['GET'])
def image():
    return render_template('bidimensional_app.html')

@app.route('/retrieval/', methods=['GET'])
def retrieval():
    return render_template('retrieval_app.html')

#RUN ON PORT 5000

if __name__ == '__main__':
    app.run(port=5000, debug=True)

print('Visit http://127.0.0.1:5000')
