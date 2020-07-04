import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import extract_image
import show_image
import pylab

def get_captcha():
    url = "http://cwcx.zju.edu.cn/WFManager/loginAction_getCheckCodeImg.action"
    for i in range(10):
        file_name = "captcha/" + str(i) + ".jpg"
        urllib.request.urlretrieve(url, file_name)
    return

data = {
        'x_train': np.zeros((40, 144)),
        'y_train': np.zeros((40, )),
    }

def get_train_data():
    for i in range(10):
        path = 'captcha/' + str(i) +".jpg"
        image = extract_image.extract_image("captcha/" + str(i) +".jpg")
        
        num = image.shape[0]
        image = image.reshape(num, 12, 12).transpose(1, 0, 2).reshape(12, num * 12)
        plt.imshow(image, cmap='gray')
        pylab.show()

        label = input("label: ")
        f = open("captcha/" + str(i) + ".txt", "w")
        f.write(label)
        f.close()

    for i in range(10):
        image_data = extract_image.extract_image('captcha/' + str(i) + '.jpg')
        data['x_train'][i*4: (i+1)*4, :] = image_data

        label = open('captcha/' + str(i) + '.txt').read()
        label = list(map(lambda x: int(x), label))
        label = np.array(label, dtype=int)
        data['y_train'][i*4: (i+1)*4] = label
        
    np.savez("hack_data.npz", x_train=data['x_train'], y_train=data['y_train'])