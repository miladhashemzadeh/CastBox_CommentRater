from Include.network import pageDownloader as pd
import re
from bs4 import BeautifulSoup as bs

dummyUrl = 'https://castbox.fm/episode/SERIAL-KILLER%3A-Bruce-McArthur-id1275406-id399043533?country=us'


class CommentExt:
    _CleanComments = []

    def _remove_html_tags(self, text):
        return re.sub(re.compile('<.*?>'), '', text)

    def mainTextExt(self, url):
        self._CleanComments = []
        self.content = pd.take_content(url)
        mainPageParsed = bs(self.content, 'html.parser')
        Comments = mainPageParsed.select('div p', {'class': 'commentItemDes'})
        for i in Comments:
            Comment = Comments.pop()
            if "commentItemDes" == Comment.get("class")[0]:
                self._CleanComments.append(self._remove_html_tags(str(Comment)))
        return self._CleanComments

    def episodeCommentTextExt(self, epUrl):
        self._CleanComments = []
        self.content = pd.take_content(epUrl)
        epParsedForComment = bs(self.content, 'html.parser')
        Comments = epParsedForComment.select('p.commentItemDes')
        for Comment in Comments:
            if "commentItemDes" == Comment.get("class")[0]:
                self._CleanComments.append(self._remove_html_tags(str(Comment)))
        return self._CleanComments


if __name__ == '__main__':
    print(CommentExt().episodeCommentTextExt(dummyUrl))
