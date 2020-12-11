## Image_Captioning
A repository for our Python/Deep Learning Group Project - an image caption generator. We used the images and captions from the [Flickr8k](https://github.com/jbrownlee/Datasets/releases) dataset to train our models.<br>

### Team Members:
Gabriella Willis, Jimmy Blundell, Shelby Mohar

### Components
There are three image-captioning models within this project. These models have several elements in common:<br>
1) All models utilize [Bahdanau's Attention](https://arxiv.org/abs/1409.0473) mechanism, though it is implemented in varying ways
2) All of the descriptions are embedding using [GloVe's pretrained embeddings](https://nlp.stanford.edu/projects/glove/) rather than a tokenizer
3) All of the image features are generated using [InceptionV3](https://keras.io/api/applications/inceptionv3/)
