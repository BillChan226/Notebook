## Object Hallucination (OH)

A typical Image captioning framework:

![image-20231029134354521](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20231029134354521.png)

### observations on OH

+ hallucinated objects often appear in the relevant scenes (e.g. “surfboard” on a beach)

+ hallucinated objects tend to be mentioned towards the end of the sentence, suggesting that some of the preceding words may have triggered hallucination

+ models with less hallucination tend to make errors consistent with the image model, whereas models with more hallucination tend to make errors consistent with the language model

+ at the beginning of training errors are more consistent with the language model, whereas towards the end of training, errors are more consistent with the image model

+ hallucinating objects impacts sentence quality not only because an object is predicted incorrectly, but also because the hallucinated word impacts generation of other words in the sentence (e.g. hallucinating a “cat” leading to hallucinating a “chair”, hallucinating a “dog” – to a “frisbee”)

  

### How to mitigate hallucination

+ **Attention**: attention lowers hallucination, but it appears that much of the gain from attention models is due to access to the underlying convolutional features as opposed the attention mechanism itself
+ **Strong Visual Processing**: models with less hallucination are better at integrating knowledge from an image into the sentence generation process
+ **Object co-occurence**: by altering the co-occurrence statistics of the objects, we lessen the models’ dependence on language prior and visual features



### Hallucination via Visual Encoder

+ **Region-based**: 
  + **output**: identify and focus on specific regions in the image provided by a high-level object detector (e.g. Faster R-CNN object detector: ResNet-101, ResNeXt152)
  + **limitation**: only represent detected objects and not the entire image. This is more severe especially with out-of-domain images
+ **Grid-based**:
  + **output**: spatial layout of an image or **feature map**  (provided by CNN, e.g. ResNet) when it's divided into equal-sized cells or subdivisions
  + **limitation**: more hallucinations due to certain **inductive biases** inherent in CNNs, such as locality and translation invariance, which may weaken the connection of different characteristics of a single object
+ **Patch-based**:
  + **output**: an image is divided into fixed-size patches. Each patch is then flattened, linearly embedded, and fed into the transformer to undergo a specific transformation or processing, like in ViT
  + **edge**: since patch-based encoders employ transformer backbone like ViT, which can inherently capture long-range dependencies and global context, which might give them an advantage over grid-based encoders that employs CNN backbone

![image-20231029195227208](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20231029195227208.png)



### Hallucination via Decoder

**decoding methods**

+ search-based: optimize for the language model log-probabilities;
+ sampling-based: draw the next token from a truncated distribution;
+ constrasting decoding: tokens only get high probability if **expert** consider them likely and **amateur** consider unlikely;

**Contrastive Decoding**

Considered a scenario involving two language models; one is a very large pre-trained model, termed the **expert**, and the other is a much smaller version of the same architecture, termed the **amateur**. Importantly, whereas these models **share some failure modes and undesirable behaviors**, the expert model clearly outperforms the amateur model in language model tasks. Thus we can exploit the contrast between the predictions of the expert and amateur to obtain an improved generated output.

![image-20231030213730967](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20231030213730967.png)

It is beneficial to prefer predictions to which only the expert model assigns a high probability, versus predictions to which both the expert and the amateur assign high probabilities. Intuitively, since the amateur model has a stronger propensity than the expert for problematic behaviors (e.g., repetitiveness in the case of text generation).

**Auto-contrastive Decoding** (ACD): both the expert and the amateur are situated in the same model, and are deﬁned by two different layers of that model.

![image-20231030232708339](/Users/zhaorunchen/Library/Application Support/typora-user-images/image-20231030232708339.png)