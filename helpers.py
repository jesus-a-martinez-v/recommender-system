import json
import re
import subprocess
import time
import xml

import mwparserfromhell
import requests
from bs4 import BeautifulSoup
from keras.utils import get_file


def get_wikipedia_dumps():
    index = requests.get('https://dumps.wikimedia.org/enwiki/').text
    soup_index = BeautifulSoup(index, 'html.parser')
    dumps = [a['href'] for a in soup_index.find_all('a')
             if a.has_attr('href') and a.text[:-1].isdigit()]

    return dumps


def download_dump(dumps):
    for dump_url in sorted(dumps, reverse=True):
        print(dump_url)
        dump_html = requests.get(f'https://dumps.wikimedia.org/enwiki/{dump_url}').text
        soup_dump = BeautifulSoup(dump_html, 'html.parser')
        pages_xml = [a['href'] for a in soup_dump.find_all('a')
                     if a.has_attr('href') and a['href'].endswith('-pages-articles.xml.bz2')]

        if pages_xml:
            break

        time.sleep(1)  # Must wait so Wikipedia does not kick us.

    wikipedia_dump = pages_xml[0].rsplit('/')[-1]
    url = f'https://dumps.wikimedia.org/{pages_xml[0]}'
    path = get_file(wikipedia_dump, url)

    return path


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        super(WikiXmlHandler, self).__init__()
        self._buffer = None
        self._values = dict()
        self._movies = list()
        self._current_tag = None

    def characters(self, content):
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attributes):
        if name in {'title', 'text'}:
            self._current_tag = name
            self._buffer = list()

    def endElement(self, name):
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            movie = self.process_article(**self._values)

            if movie:
                self._movies.append(movie)

    @staticmethod
    def process_article(title, text):
        rotten = [(re.findall('\d\d?\d?%', p), re.findall('\d\.\d\/\d+|$', p), p.lower().find('rotten-tomatoes'))
                  for p in text.split('\n\n')]

        iterator = ((perc[0], rating[0])
                    for perc, rating, index in rotten
                    if len(perc) == 1 and index > -1)
        rating = next(iterator, (None, None))

        wikicode = mwparserfromhell.parse(text)

        film = next((template for template in wikicode.filter_templates()
                     if template.name.strip().lower() == 'infobox film'), None)

        if film:
            properties = {param.name.strip_code().strip(): param.value.strip_code.strip()
                          for param in film.params
                          if param.value.strip_code().strip()}

            links = [x.title.strip_code().strip()
                     for x in wikicode.filter_wikilinks()]
            return (title, properties, links) + rating


def create_wiki_xml_parser_and_handler():
    parser = xml.sax.make_parser()
    handler = WikiXmlHandler()
    parser.setContentHandler(handler)

    return parser, handler


parser, handler = create_wiki_xml_parser_and_handler()


def parse_dumps(parser, dumps_path):
    for line in subprocess.Popen(['bzcat'], stdin=open(dumps_path), stdout=subprocess.PIPE).stdout:
        try:
            parser.feed(line)
        except StopIteration:
            break


def write_movies(handler, output_path='generated/wp_movies.ndjson'):
    with open(output_path, 'wt') as fout:
        for movie in handler._movies:
            fout.write(f'{json.dumps(movie)}\n')


if __name__ == '__main__':
    dumps = get_wikipedia_dumps()
    print(dumps)
    path = download_dump(dumps)
    parse_dumps(parser, path)
    write_movies(handler)
