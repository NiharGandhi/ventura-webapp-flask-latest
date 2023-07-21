import pickle

with open('face_enc', 'rb') as f:
    data = pickle.load(f)
    print(data)