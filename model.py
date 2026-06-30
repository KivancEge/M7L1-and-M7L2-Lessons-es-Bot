import tf_keras as keras  # tf-keras içe aktarma – .h5 uzantılı modeller ile uyumlu Keras sürümü
from tf_keras.models import load_model  # tf_keras içerisinde load_model fonksiyonunu içe aktarma, bu modeli açmamızı sağlar
import numpy as np

def predict_image(image_path,model_path,label_path):

  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model(model_path, compile=False)

  # Load the labels
  class_names = open(label_path, "r").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(image_path).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array

  # Predicts the model
  prediction = model.predict(data)
  print(prediction)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  # Print prediction and confidence score
  #print("Class:", class_name[2:], end="")
  #print("Confidence Score:", confidence_score)
  return class_name[2:], confidence_score


def get_class(model_path , labels_path , image_path ):

 # Anlaşılırlık için bilimsel gösterimin devre dışı bırakılması
    np.set_printoptions(suppress=True)

    # Model yükleniyor
    model = load_model(model_path, compile=False)

    # Etiketler yükleniyor
    class_names = open(labels_path, "r").readlines()

    # Keras modelini beslemek için doğru şekle sahip bir dizi oluşturma
    # Diziye koyabileceğiniz 'uzunluk' veya görüntü sayısı
    # Şekil (shape) tuple'ındaki ilk pozisyon tarafından belirlenir (bu örnekte: 1)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Bunu resminizin yolu ile değiştirin
    image = Image.open(image_path).convert("RGB")

    # Görüntüyü en az 224x224 piksel olacak şekilde yeniden boyutlandırma ve ardından merkezden kırpma
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Görüntüyü bir NumPy dizisine dönüştürme
    image_array = np.asarray(image)

    # Görüntüyü normalleştirme
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Görüntüyü diziye yükleme
    data[0] = normalized_image_array

    # Modelin tahmin edilmesi
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return(class_name[2:], confidence_score)
