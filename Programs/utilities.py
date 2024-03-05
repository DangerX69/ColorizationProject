from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import skimage.transform 
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import tensorflow as tf
class DataGenerator(tf.keras.utils.Sequence):

    # This class inherits from the keras sequence class. 
    # It works as the data generator but with some custom properties
    # Generates image data in batches to prevent memory overload
    # image_filenames: List of all image file names
    # batch_size: Desire batch size

    def __init__(self,image_filenames,directory,batch_size):
        'Initialization runs when this object is called upon'
        self.image_filenames = image_filenames
        self.directory = directory
        self.batch_size = batch_size
        self.epoch_end()
    
    
    def __len__(self):
        return int(np.floor(len(self.image_filenames)/self.batch_size))
    
    
    def __getitem__(self,id):
        'Generate a batch of data'
        # Generate indexes for the batch
        idx = self.indexes[id*self.batch_size:(id+1)*self.batch_size]
        
        # Find list of image_filenames
        image_filenames_temp = [self.image_filenames[i] for i in idx]
        
        # Generate data
        X,y = self.generate_process_image(image_filenames_temp)
        
        return X,y
    
    def image_process(self,image):
        '''
        
        Takes input of image in the form of a numpy array in rgb format
        provides the following Output:
            output for Encoder: size 224 x 224 x 1
            output for Extractor: size 299 x 299 x 3
            and the AB_channels: 224 x 224 x 2
        '''
        im = image.copy()#create  a copy so original remains intact for other operation
        im = rgb2lab(im)#the copy is converted to LAB
        
        # Normalization Part#############very important#####################wasted 2 days because of this
        '''
        so in the LAB color space, the values for L A and B are not the same range
        The Lightness Channel ranges from 0 - 100 for some reason
        A and B range from -127 - +127 like normal 
        
        so to normalize a range of 0 -100 into -1 to 1,  i can subtract 50 so the range is now -50 to 50,  then divide by 50 (bruh!)
        
        '''
        L_channel = im[:,:,0]-50; L_channel = L_channel/50
        
        
        A_channel = im[:,:,1]/128
        B_channel = im[:,:,2]/128
        
        #resize the L channel part to 224 by 224 for our model encoder input. anti aliasing can be set to false for sharper images
        
        Encoder_image = skimage.transform.resize(L_channel,(224,224,1),anti_aliasing = True)
        
        #transform the original image into grayscale and then resize to 299 by 299 for the inception resnet
        '''
        Also VERY IMPORTANT, the image (original) was in RGB, so we use rgb to gray function
        but but butttttt(ma pagal bhaisakein :( )
        the resnet requires a 3 channel input so we convert the gray image to rgb again
        
        natra cheating huncha because train garda colored image bata feature extract garna sikcha but practical case ma ta color hudaina
        soooo
        also also also, skimage resize use garda internally normalize huncha value so, at the end we turn the rgb values back to 0-255 range
        '''
        
        Extractor_image = rgb2gray(image)
        Extractor_image = gray2rgb(Extractor_image)
        Extractor_image = skimage.transform.resize(Extractor_image,(299,299,3),anti_aliasing = True)*255
        
        ''' Preprocess_input is from the inception resnet , need to read documentation for what it does exactly. but i trust chatgpt that it is required, will update this
        If i get time to look at it'''
        
        Extractor_image = preprocess_input(Extractor_image)
        
        
        # Self explanatory a+b =ab yay!!
        A_channel = skimage.transform.resize(A_channel,(224,224,1))
        B_channel = skimage.transform.resize(B_channel,(224,224,1))
        AB_channel = np.concatenate([A_channel,B_channel],axis=-1)
               
        return Encoder_image,Extractor_image,AB_channel     

    
    def generate_process_image(self,image_filenames_temp):
        'Generates processed image data'
        # Initialization
        X_encoder = np.empty((self.batch_size,224,224,1))
        X_extractor = np.empty((self.batch_size,299,299,3))
        y_ab_channel = np.empty((self.batch_size,224,224,2))
        
        # Store the processed image into lists
        for i , filename in enumerate(image_filenames_temp):
            # Load and process
            im=np.array(imread(os.path.join(self.directory,filename)))
            encoder,extractor,ab_channel=self.image_process(im)
            
            # Store the data
            X_encoder[i,] = encoder
            X_extractor[i,] = extractor
            y_ab_channel[i,] = ab_channel
            
        return [X_encoder,X_extractor],y_ab_channel
    
    def epoch_end(self):
        'Updates the indexes when each epoch end'
        self.indexes = np.arange(len(self.image_filenames))
     
'''making the method definition as a seperate function first to test and it worked!!'''       
def process(image):

        im = image.copy()
        im = rgb2lab(im)
        
  
        L_channel = im[:,:,0]-50; L_channel = L_channel/50
        A_channel = im[:,:,1]/128
        B_channel = im[:,:,2]/128
        
      
        Encoder_image = skimage.transform.resize(L_channel,(224,224,1),anti_aliasing = True)
        
      
        Extractor_image = rgb2gray(image)
        Extractor_image = gray2rgb(Extractor_image)
        Extractor_image = skimage.transform.resize(Extractor_image,(299,299,3),anti_aliasing = True)*255
        Extractor_image = preprocess_input(Extractor_image)
        
        
       
        A_channel = skimage.transform.resize(A_channel,(224,224,1))
        B_channel = skimage.transform.resize(B_channel,(224,224,1))
        AB_channel = np.concatenate([A_channel,B_channel],axis=-1)
               
        return Encoder_image,Extractor_image,AB_channel