import pickle

# -------------------------------------------------
# load_dict_data
# -------------------------------------------------
def load_dict_data(pickle_path = '../finanzen.net.pickle'):
    
    ret = None
    with open(pickle_path, 'rb') as f_in:
        ret = pickle.loads(f_in.read())

    return ret