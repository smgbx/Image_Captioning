## Image_Captioning
A repository for our Python/Deep Learning Group Project - an image caption generator. We used the images and captions from the [Flickr8k](https://github.com/jbrownlee/Datasets/releases) dataset to train our models.<br>

### Team Members:
Gabriella Willis, Jimmy Blundell, Shelby Mohar

### Components
#### Image Captioning Models
There are three image-captioning models within this project. These models have several elements in common:<br>
1) All models utilize [Bahdanau's Attention](https://arxiv.org/abs/1409.0473) mechanism, though it is implemented in varying ways<br>
2) All of the descriptions are embedding using [GloVe's pretrained embeddings](https://nlp.stanford.edu/projects/glove/) rather than a tokenizer<br>
3) All of the image features are generated using [InceptionV3](https://keras.io/api/applications/inceptionv3/)<br>
4) All of the models follow the [merge](https://arxiv.org/abs/1708.02043) model. This means that the RNN is associated only with textual data which is then later feed into a seperate decoder, rather than the RNN serving as a decorder for both the image and textual data <br>
![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Schematic-of-the-Merge-Model-For-Image-Captioning.png)<br>

The models differ in the ways in which the attention mechanism is implemented. Each model and associated files are seperated into folders:<br>
[Model I - Bahdanau Attention for Images with GloVe](https://github.com/smgbx/Image_Captioning/tree/main/Model%20I%20-%20Bahdanau%20Attention%20for%20Images%20with%20GloVe) - The attention for this model is focused only on the image features. <br>
[Model II - Bahdanau Attention for Text with GloVe](https://github.com/smgbx/Image_Captioning/tree/main/Model%20II%20-%20Bahdanau%20Attention%20for%20Text%20with%20GloVe) - The attention for this model is focused only on the text. <br>
[Model III - Bahdanau Attention for Images and Text with GloVe](https://github.com/smgbx/Image_Captioning/tree/main/Model%20III%20-%20Bahdanau%20Attention%20for%20Images%20and%20Text%20with%20GloVe) - This model implements attention for both the text and the image features components.<br>

#### Preparation
There are two helper notebooks used for preparing the data:<br>
[GetFlickerImagesFeatures](https://github.com/smgbx/Image_Captioning/blob/main/GetFlickerImagesFeatures.ipynb) - This notebook is used for generating the image features from the dataset using a modified version of the InceptionV3 image classification model. These features are then saved to a file and are be used to train the model. <br>
[CleanFlickerDescriptions](https://github.com/smgbx/Image_Captioning/blob/main/CleanFlickerDescriptions.ipynb) - This notebook is used to clean and preprocess the list of descriptions. These descriptions are loading into a dictionary in which the key is the image id, and the value is a list of associated captions.<br>

#### Evaluation
There are two helper notebooks used for evaluating the performance of the models:<br>
[EvaluateModel](https://github.com/smgbx/Image_Captioning/blob/main/EvaluateModel.ipynb) - This notebook is uses two metrics to evaluate the models: the BLEU score, and the METEOR score. It plots the loss/val_loss of the model and demonstrates the captioning ability of the model on images from the test set.<br>
[CaptioningNewImages](https://github.com/smgbx/Image_Captioning/blob/main/CaptioningNewImages.ipynb) - This notebook allows the user to caption uploaded images.<br>

### Resources:
[Where to put the Image in an Image Caption Generator](https://arxiv.org/abs/1703.09137)<br>
[Image captioning with visual attention](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb)<br>
[How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)<br>
[Create your Own Image Caption Generator using Keras!](https://www.analyticsvidhya.com/blog/2020/11/create-your-own-image-caption-generator-using-keras/)<br>
[Neural Machine Translation using Bahdanau Attention Mechanism](https://medium.com/analytics-vidhya/neural-machine-translation-using-bahdanau-attention-mechanism-d496c9be30c3)<br>
[A Comprehensive Guide to Attention Mechanism in Deep Learning for Everyone](https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/)<br>
[How to add attention layer to a Bi-LSTM](https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137)<br>
[How to add Attention on top of a Recurrent Layer (Text Classification) #4962](https://github.com/keras-team/keras/issues/4962)<br>

