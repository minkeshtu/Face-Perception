'''
Loading the Affetnet dataset 'https://arxiv.org/pdf/1708.03985.pdf' into the pandas dataframe.
'''

import pandas as pd 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt
import os

expression_list = {
    '0': 'Neutral',
    '1': 'Happy',
    '2': 'Sad',
    '3': 'Surprise',
    '4': 'Fear',
    '5': 'Disgust',
    '6': 'Anger',
    '7': 'Contempt',
    '8': 'None',
    '9': 'Uncertain',
    '10': 'Non-Face'
    }

def index_to_exp(col_value):
    return expression_list[str(col_value)] 

ROOT_DIR = os.path.abspath('')
dataset_dir = os.path.join(ROOT_DIR, 'datasets/Affectnet/')


train_dataframe = pd.read_csv(f'{dataset_dir}Manually_Annotated_file_lists/training.csv' , usecols=['subDirectory_filePath', 'expression'], converters={
    'expression': index_to_exp
})

val_dataframe = pd.read_csv(f'{dataset_dir}Manually_Annotated_file_lists/validation.csv', usecols=[0, 6], header=None, converters={
    6: index_to_exp
}, names=['subDirectory_filePath', 'expression'])


'''
print(val_dataframe.head())
print(train_dataframe.info())
print(val_dataframe.info())
'''


'''
def check_file(col_value):
    path = 'Manually_Annotated_Images/' + col_value
    if os.path.isfile(path=path):
        return col_value
    else:
        return 'NaN'

def ran_num():
    return np.random.randint(0, 414799)

root_dir_for_images = "Manually_Annotated_Images/"

figure = plt.Figure()
for i in range(10):
    plt.subplot(5, 5, i+1)
    r_num = ran_num()
    img = cv.imread(f"{root_dir_for_images}"+f"{dataframe.subDirectory_filePath[r_num]}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    expression_v = f"{dataframe.expression[r_num]}"
    plt.xlabel(f"{expression_list[expression_v]}")

plt.show()
key = cv.waitKey(0)
if key == ord('q'):
    cv.destroyAllWindows()
    
'''