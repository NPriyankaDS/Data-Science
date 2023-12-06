from keras.models import load_model
def model_load():
    model = load_model('tomatos.h5')
    return model
