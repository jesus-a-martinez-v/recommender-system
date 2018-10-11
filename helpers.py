import json
import random
import re
import subprocess
import time
import xml.sax
from collections import Counter

import mwparserfromhell
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras import Input
from keras.layers import Embedding, Dot, Reshape
from keras.models import Model
from keras.utils import get_file
from sklearn import svm
from sklearn.linear_model import LinearRegression


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
    # dumps = get_wikipedia_dumps()
    # print(dumps)
    # path = download_dump(dumps)
    # parse_dumps(parser, path)
    # write_movies(handler)

    with open('data/wp_movies_10k.ndjson') as fin:
        movies = [json.loads(line) for line in fin]

    link_counts = Counter()

    for movie in movies:
        movie_links = movie[2]
        link_counts.update(movie_links)

    print(link_counts.most_common(10))

    top_links = [link for link, count in link_counts.items() if count >= 3]
    link_to_index = {link: index for index, link in enumerate(top_links)}
    movie_to_index = {movie[0]: index for index, movie in enumerate(movies)}

    pairs = []
    for movie in movies:
        movie_title = movie[0]
        movie_links = movie[2]
        pairs.extend((link_to_index[link], movie_to_index[movie_title])
                     for link in movie_links
                     if link in link_to_index)

    pairs_set = set(pairs)
    print(len(pairs))
    print(len(top_links))
    print(len(movie_to_index))


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


    model = get_movie_embedding_model(top_links, movie_to_index)
    model.summary()

    random.seed(5)


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


    positive_samples_per_batch = 512

    model.fit_generator(batchifier(pairs, pairs_set, top_links, movie_to_index, positive_samples_per_batch),
                        epochs=15,
                        steps_per_epoch=len(pairs),
                        verbose=2)

    movie = model.get_layer('movie_embedding')
    movie_weights = movie.get_weights()[0]
    movie_lengths = np.linalg.norm(movie_weights, axis=1)
    normalized_movies = (movie_weights.T / movie_lengths).T


    def similar_movies(movie, movies, normalized_movies, movie_to_index, top_n=10):
        distances = np.dot(normalized_movies, normalized_movies[movie_to_index[movie]])
        closest = np.argsort(distances)[-top_n:]

        for c in reversed(closest):
            movie_title = movies[c][0]
            distance = distances[c]
            print(c, movie_title, distance)


    similar_movies('Rogue One', movies, normalized_movies, movie_to_index)

    link = model.get_layer('link_embedding')
    link_weights = link.get_weights()[0]
    link_lengths = np.linalg.norm(link_weights, axis=1)
    normalized_links = (link_weights.T / link_lengths).T


    def similar_links(link, top_links, normalized_links, link_to_index, top_n=10):
        distances = np.dot(normalized_links, normalized_links[link_to_index[link]])
        closest = np.argsort(distances)[-top_n:]

        for l in reversed(closest):
            distance = distances[l]
            print(l, top_links[l], distance)


    similar_links('George Lucas', top_links, normalized_links, link_to_index)

    best = ['Star Wars: The Force Awakens', 'The Martian (film)', 'Tangerine (film)', 'Straight Outta Compton (film)',
            'Brooklyn (film)', 'Carol (film)', 'Spotlight (film)']
    worst = ['American Ultra', 'The Cobbler (2014 film)', 'Entourage (film)', 'Fantastic Four (2015 film)',
             'Get Hard', 'Hot Pursuit (2015 film)', 'Mortdecai (film)', 'Serena (2014 film)', 'Vacation (2015 film)']

    y = np.asarray(([1] * len(best)) + ([0] * len(worst)))
    X = np.asarray([normalized_movies[movie_to_index[movie]]
                    for movie in (best + worst)])

    print(X.shape)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X, y)

    estimated_movie_ratings = classifier.decision_function(normalized_movies)
    best = np.argsort(estimated_movie_ratings)
    print('Best: ')
    for c in reversed(best[-5:]):
        print(c, movies[c][0], estimated_movie_ratings[c])

    print('Worst: ')
    for c in best[:5]:
        print(c, movies[c][0], estimated_movie_ratings[c])

    rotten_y = np.asarray([float(movie[-2][:-1]) / 100 for movie in movies if movie[-2]])
    rotten_X = np.asarray([normalized_movies[movie_to_index[movie[0]]] for movie in movies if movie[-2]])

    TRAINING_CUT_OFF = int(len(rotten_X) * 0.8)
    regressor = LinearRegression()
    regressor.fit(rotten_X[:TRAINING_CUT_OFF], rotten_y[:TRAINING_CUT_OFF])

    error = (regressor.predict(rotten_X[TRAINING_CUT_OFF:]) - rotten_y[TRAINING_CUT_OFF:])
    print(f'Mean Squared Error: {np.mean(error ** 2)}')

    error = (regressor.mean(rotten_y[:TRAINING_CUT_OFF]) - rotten_y[TRAINING_CUT_OFF:])
    print(f'Mean Squared Error: {np.mean(error ** 2)}')


    def gross(movie):
        v = movie[1].get('gross')

        if not v or ' ' not in v:
            return None

        v, unit = v.split(' ', 1)
        unit = unit.lower()

        if unit not in {'million', 'billion'}:
            return None

        if not v.startswith('$'):
            return None

        try:
            v = float(v[1:])
        except ValueError:
            return None

        if unit == 'billion':
            v *= 1000

        return v

    movie_gross = [gross(movie) for movie in movies]
    movie_gross = np.asarray([g for g in movie_gross if g is not None])

    highest = np.argsort(movie_gross)[-10:]

    for c in reversed(highest):
        print(c, movies[c][0], movie_gross[c])

    gross_y = np.asarray([g for g in movie_gross if g])
    gross_X = np.asarray([normalized_movies[movie_to_index[movie[0]]] for movie, g in zip(movies, movie_gross) if g])

    TRAINING_CUT_OFF = int(len(gross_X) * 0.8)
    r = LinearRegression()
    r.fit(gross_X[:TRAINING_CUT_OFF], gross_y[:TRAINING_CUT_OFF])

    error = (r.predict(gross_X[TRAINING_CUT_OFF:]) - gross_y[TRAINING_CUT_OFF:])
    print(f'Mean Squared Error: {np.mean(error ** 2)}')

    error = (np.mean(gross_y[:TRAINING_CUT_OFF]) - gross_y[TRAINING_CUT_OFF:])
    print(f'Mean Squared Error: {np.mean(error ** 2)}')