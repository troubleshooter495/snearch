import telebot
import python_files.classifier as classifier
import python_files.model as model
import cv2
import torch
import requests
import python_files.knn as knn
from python_files.model import AE3
from python_files.classifier import Classifier
from tok import token
from python_files.paths import img_paths
from torchvision.transforms import transforms

TOKEN = token()

bot = telebot.TeleBot(TOKEN, parse_mode=None)
Model = model.load_model('model.pt')
Classy = classifier.load_classifier('classifier.pt')
img_paths = img_paths('../data/ut-zap50k-images')
# samples, labels = knn.knn_samples('data/meta-data.csv', img_paths, Model)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('not shoe', 'shoe')


@bot.message_handler(commands=['start', 'help'])
def welcome(message):
    bot.reply_to(message, "Отправь мне фотографию обуви и я пришлю похожие")


@bot.message_handler(content_types=['photo'])
def reply(message):
    path = bot.get_file(message.photo[-1].file_id).file_path
    response = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{path}')

    file = open(f"../pics/photo{message.id}.png", "wb")
    file.write(response.content)
    file.close()

    img = cv2.imread(f"../pics/photo{message.id}.png")
    res = transform(cv2.resize(img, (32, 32)))
    pred = torch.argmax(Classy(res.view(1, 3, 32, 32)))
    bot.send_message(message.chat.id, classes[int(pred)])


bot.polling()
