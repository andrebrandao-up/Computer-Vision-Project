import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
os.environ['GLOG_minloglevel'] = '3'
import caffe
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

mean_filename='models/mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

age_net_pretrained='models/age_net.caffemodel'
age_net_model_file='age_net_definitions/deploy.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(im,vmin=0, vmax=0.3)
    plt.savefig("conv2_output.png")

def get_age(image_path):
    input_image = caffe.io.load_image(image_path)
    #input_image = image_path
    prediction = age_net.predict([input_image])
    return age_list[prediction[0].argmax()]

#if __name__ == "__main__":
#    print(get_age('../AgeGenderDeepLearning/erica_face.jpg'))
    
