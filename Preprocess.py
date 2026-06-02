# a class making different HRTF preprocessing algorithms available

class Preprocess:
    
    def __init__(self):
        self.algorithms = ['MagLS']
        self.current = 'MagLS'

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, algorithm='MagLS'):
        match algorithm:
            case 'MagLS':
                print("Using MagLS HRTF Preprocessing")
                return self.__mag_ls(hrirs)

    def __mag_ls(self, hrirs):
        return hrirs