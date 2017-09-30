import pickle #saving and loading our model
import numpy as np
from app.model.preprocessor import Processor as image_prep

#class for loading our saved model and classifying new images
class MyOCR:
    #init method
    def __init__(self, filename='my_wts.pkl',pool_size=2):
        [weights, meta] = pickle.load(open(filename,'rb'),encoding='latin1')

        #list to store labels
        self.vocab = meta['vocab']

        #get rows and columns in image
        self.img_rows = meta['img_side']
        self.img_cols = meta['img_side']

        #load our model
        self.CNN = MyCNN()
        self.CNN.load_weights = weights
        self.CNN.pool_size = int(pool_size)

    #classify new image
    def predict(self,image):
        print(image.shape)

        #vectorize the image ready to feed into network
        X = np.reshape(image, (1,1,self.img_rows,self.img_cols))
        X = X.astype('float32')

        #predict
        predicted = self.CNN.predict(X)

        #return the predicted label
        return self.vocab[predicted]
class CNN():
    def __init__(self):
        #store layers ie. weights for each layer that we learnt
        self.layers = []

        #window size for pooling
        self.pool_size = None

    def load_weights(self,weights):
        assert not self.layers,"Weights can be loaded once"

        #add the saved weight matrix to convolution network
        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])

    def predict(self,X):
        h = self.cnn_layer(X, layer_i = 0, border_mode="full"); X = h;
        h = self.relu_layer(X);X = h #non_linearity ie. activation


        h = self.cnn_layer(X, layer_i = 2, border_mode="valid"); X = h;
        h = self.relu_layer(X); X = h

        h = self.maxpool_layer(X); X = h

        h = self.dropout_layer(X, 0.25); X = h
        #h = self.flatten_layer(X, layer_i = 7); X= h
        h = self.flatten_layer(X); X= h

        h = self.fully_connected_layer(X, layer_i = 10)
        X = h
        h = self.softmax_layer2D(X); X = h

        max_i  = self.classify(X)
        return max_i[0]

    def cnn_layer(self, X, layer_i = 0, border_mode="full"):

        # store feature maps and bias  values
        features = self.layers[layer_i]["param_0"]
        bias =self.layers[layer_i]["param_1"]

        #get the filter/patch dimension
        patch_dim = features[0].shape[-1]

        no_features = features.shape[0]
        img_dim = X.shape[2]  #assuming img is square shaped

        #R G B values
        img_channels = X.shape[1]

        no_imgs = X.shape[0]

        if border_mode == "full":
            conv_dim = img_dim + patch_dim - 1
        elif border_mode == "valid":
            conv_dim = img_dim - patch_dim + 1

        #initiate feature matrix ie. weight matrix
        convolved_features = np.zeros((no_imgs,no_features,conv_dim ,conv_dim))

        #iterate through each img in training
        for img_i in range(no_imgs):
            #for each feature
            for feature_i in range(no_features):
                #first initialize convolved image
                convolved_img = np.zeros((conv_dim,conv_dim))

                #now for each channel
                for channel in range(img_channels):

                    #extract a feature from feature map
                    feature = features[feature_i, channel, :, :]

                    #define a channel specific part of our image
                    image = X[img_i, channel, :, :]

                    #now perform convolution on our img using give feature
                    convolved_img += self.convolve2D(image,feature,border_mode);

                #now add bias to our convolved_img
                convolved_img = convolved_img + bias[feature_i]

                #and add it to our list of convolved_features
                convolved_features[img_i,feature_i,:,:] = convolved_img
        return convolved_features


    @staticmethod
    def convolve2D(image,feature,border_mode="full"):
        #define the tensor dimensions of image and feature
        img_dim =np.array(image.shape)
        feature_dim = np.array(feature.shape)

        target_dim = img_dim + feature_dim - 1


        fft_result = np.fft.fft2(image,target_dim) * np.fft.fft2(feature,target_dim)
        target = np.fft.ifft2(fft_result).real

        if border_mode == "valid":
            #decide a target dimension to convlolve around
            valid_dim = img_dim - feature_dim + 1
            if np.any(valid_dim<1):
                valid_dim = feature_dim - img_dim + 1
            start_i = (target_dim - valid_dim) //2
            end_i = start_i + valid_dim
            target = target[start_i[0]:end_i[0],start_i[1]:end_i[1]]
        return target

    def relu_layer(self,X):
        z = np.zeros_like(X)
        return np.where(X>z,X,z)

    def maxpool_layer(self,convolved_features):
        #given learned features and images
        no_features = convolved_features.shape[0]
        no_img = convolved_features.shape[1]
        conv_dim = convolved_features.shape[2]
        res_dim = int(conv_dim/self.pool_size)

        #initialize our feature list as empty
        pooled_features = np.zeros((no_features,no_img,res_dim,res_dim))

        #iterate over each image
        for image_i in range(no_img):
            #for each feature map
            for feature_i in range(no_features):
                #no begin by row
                for pool_row in range(res_dim):
                    #define start and end points
                    row_start = pool_row*self.pool_size
                    row_end = row_start + self.pool_size

                    #now for each column
                    for pool_col in range(res_dim):
                        #define start and end points
                        col_start = pool_col * self.pool_size
                        col_end = col_start + self.pool_size

                        #now define a patch given start and end pints
                        patch = convolved_features[feature_i,image_i,row_start:row_end,col_start:col_end]

                        #now take the max value from that patch
                        pooled_features[feature_i,image_i,pool_row,pool_col] = np.max(patch)
        return pooled_features
    def softmax_layer2D(w):
        maxes = np.amax(w ,axis=1)
        maxes = maxes.reshape(maxes.shape[0],1)
        e = np.exp(w-maxes)
        dist =  e / np.sum(e,axis=1,keepdims=True)
        return dist

    def dropout_layer(self,X,p):
        retain_prob = 1.0- p
        X *= retain_prob
        return X

    #now flatten our output into a vector
    def flatten_layer(self,X):
        flatX = np.zeros((X.shape[0],np.prod(X.shape[1:])))
        for i in range(X.shape[0]):
            flatX[i,:] = X[i].flatten(order="C")
        return flatX
    def fully_connected_layer(self,X, layer_i = 0):
        W = self.layers[layer_i]["param_0"]
        b = self.layers[layer_i]["param_1"]
        output = np.dot(X,W) + b
        return output



    def classify(self,X):
        return X.argmax(axis=-1)








