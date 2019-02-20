import pickle
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


model = load_model('model')
token = Tokenizer()
with open('tokenizer.pickle', 'rb') as file:
    token = pickle.load(file)
seq_length = 200

def predict(text):
    text = [text]
    text = token.texts_to_sequences(text)
    text = sequence.pad_sequences(text, maxlen=seq_length, padding='post')
    predict_prob = model.predict(text)[0][0]
    predict_class = model.predict_classes(text)[0][0]
    predict_senti = 'neg' if predict_class == 1 else 'pos'
    return predict_prob, predict_class, predict_senti


# neg review
text = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in."    
# pos review
text2 = "BLACK WATER is a thriller that manages to completely transcend it’s limitations (it’s an indie flick) by continually subverting expectations to emerge as an intense experience.In the tradition of all good animal centered thrillers ie Jaws, The Edge, the original Cat People, the directors know that restraint and what isn’t shown are the best ways to pack a punch. The performances are real and gripping, the crocdodile is extremely well done, indeed if the Black Water website is to be believed that’s because they used real crocs and the swamp location is fabulous.If you are after a B-grade gore fest croc romp forget Black Water but if you want a clever, suspenseful ride that will have you fearing the water and wondering what the hell would I do if i was up that tree then it’s a must see."

predict_prob, predict_class, predict_senti = predict(text2)
print('predict prob:->>', predict_prob)
print('predict class:->>', predict_class)
print('predict senti:->>', predict_senti)
