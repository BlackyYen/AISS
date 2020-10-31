import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

#load the model
#load json and create model
json_file = open("../weights/mobilenetv2_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("../weights/mobilenetv2_final_weights.h5")

# 輸入讀取路徑
img_path = "test3.jpg"

# opencv讀檔
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# image讀檔
# img_path = "data/test/bird/1.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# img = np.array(img)
# img = np.expand_dims(img, axis=0)
# img = preprocess_input(img)

# 預測
preds = model.predict(img)
predictions = np.argmax(preds, axis = 1)[0]
classes = ['amorphous', 'normal', 'pyriform', 'tapered']
print('Predicted:', classes[predictions])
for i in range(len(classes)):
    print(classes[i], ':',round(preds[0][i], 2)*100, '%')

#Test set
# test_datagen= ImageDataGenerator(preprocessing_function=preprocess_input)
# test_generator=test_datagen.flow_from_directory('../dataset/sperm/test/',
#                                          target_size=(224,224),
#                                          color_mode='rgb',
#                                          batch_size=64,
#                                          class_mode='categorical',
#                                          shuffle=False)



# prepares data for confusion matrix (擇一)
# Y_pred = model.predict_generator(test_generator, test_generator.n//test_generator.batch_size+1)
# Y_pred = model.predict_generator(test_generator, test_generator.n//test_generator.batch_size)

# 輸出混淆矩陣與分類數據
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# target_names = ['amorphous', 'normal', 'pyriform', 'tapered']
# print('Classification Report')
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))


