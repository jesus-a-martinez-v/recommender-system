import json
import random
import re
import subprocess
import time
import xml.sax

import mwparserfromhell
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras import Input
from keras.layers import Embedding, Dot, Reshape
from keras.models import Model
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


def get_movie_embedding_model(top_links, movie_to_index, embedding_size=50):
    link = Input(name='link', shape=(1,))
    movie = Input(name='movie', shape=(1,))

    link_embedding = Embedding(name='link_embedding',
                               input_dim=len(top_links),
                               output_dim=embedding_size)(link)
    movie_embedding = Embedding(name='movie_embedding',
                                input_dim=len(movie_to_index),
                                output_dim=embedding_size)(movie)
    dot = Dot(name='dot_product',
              normalize=True,
              axes=2)([link_embedding, movie_embedding])

    merged = Reshape(target_shape=(1,))(dot)

    model = Model(inputs=[link, movie], outputs=[merged])
    model.compile(optimizer='nadam', loss='mse')

    return model


def batchifier(pairs, pairs_set, top_links, movie_to_index, positive_samples=50, negative_ratio=10):
    batch_size = positive_samples * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))

    while True:
        for index, (link_id, movie_id) in enumerate(random.sample(pairs, positive_samples)):
            batch[index, :] = (link_id, movie_id, 1)

        index = positive_samples

        while index < batch_size:
            movie_id = random.randrange(len(movie_to_index))
            link_id = random.randrange(len(top_links))

            if not (link_id, movie_id) in pairs_set:
                batch[index, :] = (link_id, movie_id, -1)
                index += 1

        np.random.shuffle(batch)

        yield {'link': batch[:, 0],
               'movie': batch[:, 1]}, batch[:, 2]


def similar_movies(movie, movies, normalized_movies, movie_to_index, top_n=10):
    distances = np.dot(normalized_movies, normalized_movies[movie_to_index[movie]])
    closest = np.argsort(distances)[-top_n:]

    for c in reversed(closest):
        movie_title = movies[c][0]
        distance = distances[c]
        print(c, movie_title, distance)


def similar_links(link, top_links, normalized_links, link_to_index, top_n=10):
    distances = np.dot(normalized_links, normalized_links[link_to_index[link]])
    closest = np.argsort(distances)[-top_n:]

    for l in reversed(closest):
        distance = distances[l]
        print(l, top_links[l], distance)
