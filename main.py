import Include.scrapper.CBCommentScrapper as cb
from keras.models import load_model
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    dummyUrlEP = 'https://castbox.fm/episode/%DA%AF%D9%84-%D8%AF%D9%88-%D8%B1%D9%88%DB%8C-%3A-%D9%82%D8%B5%DB%8C%D8%AF%D9%87-%D8%B3%D8%B9%D8%AF%DB%8C-%D9%88-%D8%B3%D8%B1%D9%88%D8%B4-%D8%A7%D8%B5%D9%81%D9%87%D8%A7%D9%86%DB%8C-%D9%88-%D9%81%D8%B1%D8%AE%DB%8C-%D8%B3%DB%8C%D8%B3%D8%AA%D8%A7%D9%86%DB%8C-id3935515-id379893136'
    dummyUrlMain = 'https://castbox.fm/channel/id3935515?utm_campaign=ex_share_ch&utm_medium=exlink&country=us'
    model = load_model('C:/Users/VampyreLord/PycharmProjects/CastBoxcomment_rate/Include/ML/persian-sentiment-fasttext.model')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model.summary()
    comments = cb.CommentExt().episodeCommentTextExt(dummyUrlEP)
    for comment in comments:
        print(model.predict(comment))

