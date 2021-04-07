import telebot
import random
import model
from model import AE3
import knn
import dataset
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

TOKEN = ""


class STATE:
    states = ['turned_off', 'reminded', 'waiting', 'got_num']

    def __init__(self):
        self.state = self.states[0]

    def update(self, new_state):
        self.state = new_state


st = STATE()
img_folder = 'data/ut-zap50k-images'
bot = telebot.TeleBot(TOKEN, parse_mode=None)

Model = model.load_model()
ds = dataset.ShoesDataset('data/meta-data.csv', img_folder,
                          lambda: random.randint(1, 100) > 50)
samples, labels, img_paths = knn.knn_samples('data/meta-data.csv', ds.img_paths,
                                             Model)
ids, dists = knn.count_distances(samples, 5)

messages = {}  # chat id + msg id = img index
user_likes = {}  # user id = img index
user_dislikes = {}  # user id = img index
user_used = {}  # user id = img index


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    st.update('waiting')
    bot.reply_to(message,
                 "I will send you pictures of shoes, you can rate them "
                 "and according to your ratings I will show you more of what "
                 "you like 🙂\n If you want to reset your ratings, you should "
                 "run /reset command")

    if message.chat.id not in user_likes:
        user_likes[message.from_user.id] = set()
        user_dislikes[message.from_user.id] = set()
        user_used[message.from_user.id] = set()

    i = random.randint(0, len(img_paths))
    img = open(img_paths[i], 'rb')
    r = bot.send_photo(message.chat.id, img, reply_markup=gen_markup())
    messages[str(r.chat.id) + str(r.id)] = i
    user_used[message.from_user.id].add(i)


def gen_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("👍🏻", callback_data="like"),
               InlineKeyboardButton("👎🏿", callback_data="dislike"))
    return markup


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.from_user.id not in user_likes:
        print(user_likes)
        print(call.from_user.id)
        return

    if call.data == "like":
        bot.answer_callback_query(call.id, "like")
        user_likes[call.from_user.id].add(
            messages[str(call.message.chat.id) + str(call.message.id)])
        if len(user_likes[call.from_user.id]) < 5:
            for j in ids[messages[str(call.message.chat.id) +
                                  str(call.message.id)]]:
                if j not in user_dislikes[call.from_user.id] and \
                        j not in user_used[call.from_user.id]:
                    user_likes[call.from_user.id].add(j)
    elif call.data == "dislike":
        user_dislikes[call.from_user.id].add(
            messages[str(call.message.chat.id) + str(call.message.id)])
        bot.answer_callback_query(call.id, "dislike")

    if user_likes[call.from_user.id]:
        i = random.choice(tuple(user_likes[call.from_user.id]))
        user_likes[call.from_user.id].remove(i)

        while i in user_dislikes[call.from_user.id] or \
                i in user_used[call.from_user.id]:
            if user_likes[call.from_user.id]:
                i = random.choice(tuple(user_likes[call.from_user.id]))
                user_likes[call.from_user.id].remove(i)
            else:
                i = random.randint(0, len(img_paths))

        print(user_likes[call.from_user.id])
    else:
        i = random.randint(0, len(img_paths))
        while i in user_dislikes[call.from_user.id] or \
                i in user_used[call.from_user.id]:
            i = random.randint(0, len(img_paths))

    img = open(img_paths[i], 'rb')
    r = bot.send_photo(call.message.chat.id, img, reply_markup=gen_markup())
    messages[str(r.chat.id) + str(r.id)] = i

    print(i, i in user_likes[call.from_user.id],
          i in user_dislikes[call.from_user.id],
          i in user_used[call.from_user.id])

    user_used[call.from_user.id].add(i)


@bot.message_handler(commands=['reset'])
def reset_likes(message):
    user_likes[message.from_user.id] = set()
    user_dislikes[message.from_user.id] = set()
    user_used[message.from_user.id] = set()

    bot.send_message(message.chat.id, "Ratings are reset!")

    i = random.randint(0, len(img_paths))
    img = open(img_paths[i], 'rb')
    r = bot.send_photo(message.chat.id, img, reply_markup=gen_markup())
    messages[str(r.chat.id) + str(r.id)] = i
    user_used[message.from_user.id].add(i)


@bot.message_handler(func=lambda x: st.state == 'turned_off')
def turned_off_msg(message):
    bot.send_message(message.chat.id, 'You should /start me first...')
    st.update('reminded')


bot.polling()
