from predicting.predicting import predict
from PIL import Image

image = Image.open('predicting/face2.jpeg')
image.thumbnail((360, 360))
image.save('image_400.jpg')

# predict("predicting/face.jpg", "predicting/University_ITMO.mp3", model_source="predicting/weights-2.h5", only_weights=True)
predict(
    "predicting/face.jpg",
    "predicting/University_ITMO.mp3",
    model_source="predicting/TimeDistributedModel/TimeDistributedModel",
    only_weights=True
)
