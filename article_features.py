import os
import sys
import shlex
import csv

from subprocess import run
from collections import defaultdict, Counter

_SPLIT_PREFIX = 'ARTICLE_ID_'

_booknlp_cmd = ('./runjava novels/BookNLP '
                '-doc {} '
                '-p data/temp '
                '-tok {}.tokens '
                '-f -q')
def booknlp_process(filename):
    outfile, ext = os.path.splitext(filename)
    cmd = _booknlp_cmd.format(filename, outfile)
    run(shlex.split(cmd))
    return outfile + '.tokens'

def load_article_tokens(token_filename):
    with open(token_filename, encoding='utf-8') as ip:
        rd = csv.DictReader(ip, delimiter='\t', quoting=csv.QUOTE_NONE)
        tokens = list(rd)

    articles = {}
    article = []
    for t in tokens:
        # The first token should be an article id / split point. If not,
        # any lines preceding the first artice id will be ignored.
        if t['originalWord'].startswith(_SPLIT_PREFIX):
            aid = int(t['originalWord'].replace(_SPLIT_PREFIX, ''))
            article = []
            articles[aid] = article
            continue
        else:
            article.append(t)
    return articles

def load_metadata(metadata_filename):
    with open(metadata_filename, encoding='utf-8') as ip:
        return list(csv.DictReader(ip))

def concat_input(metadata, data_folder, concat_filename):
    filenames = [os.path.join(data_folder, r['Filename']) for r in metadata]
    with open(concat_filename, 'w', encoding='utf-8') as op:
        for i, fn in enumerate(filenames):
            op.write('\n\n{}{}\n\n'.format(_SPLIT_PREFIX, i))
            with open(fn, encoding='utf-8') as ip:
                op.write(ip.read())

def metadata_tokens_join(metadata, tokens):
    return [mk_article(md, tokens[i]) for i, md in enumerate(metadata)]

def mk_article(metadata, tokens):
    return {'article_id': metadata['Filename'],
            'author': metadata['Author'],
            'date': metadata['Date'],
            'tokens': tokens}

def article_features(article):
    article_id = article['article_id']
    author = article['author']
    date = article['date']
    features = aggregate_tokens(article['tokens'])
    for f_type, f_values in features.items():
        for fv, count in f_values.items():
            yield {
                'article_id': article_id,
                'author': author,
                'date': date,
                'feature_type': f_type,
                'feature': fv,
                'count': count
            }

def skip_entity(tokens, tix, entity_key):
    ent = tokens[tix][entity_key]
    while tix < len(tokens) and tokens[tix][entity_key] == ent:
        tix += 1
    return tix

def aggregate_tokens(tokens):
    features = defaultdict(Counter)
    tix = 0
    while tix < len(tokens):
        token = tokens[tix]
        if token['ner'] == 'ORGANIZATION':
            features['org'][token['entityName']] += 1
            tix = skip_entity(tokens, tix, 'entityName')
        elif token['ner'] == 'LOCATION':
            features['loc'][token['entityName']] += 1
            tix = skip_entity(tokens, tix, 'entityName')
        elif token['characterName']:
            features['person'][token['characterName']] += 1
            tix = skip_entity(tokens, tix, 'characterName')
        elif token['attributionName']:
            features['source'][token['attributionName']] += 1
            tix = skip_entity(tokens, tix, 'attributionName')
        else:
            tix = tix + 1
    return features


if __name__ == '__main__':
    metadata_filename = sys.argv[1]
    data_folder = sys.argv[2]
    output_file = sys.argv[3]
    input_file, _ext = os.path.splitext(metadata_filename)
    input_file += '-concat.txt'

    metadata = load_metadata(metadata_filename)
    concat_input(metadata, data_folder, input_file)
    tokenfile = booknlp_process(input_file)
    article_tokens = load_article_tokens(tokenfile)
    articles = metadata_tokens_join(metadata, article_tokens)
    features = [feature
                for article in articles
                for feature in article_features(article)]

    with open(output_file, 'w', encoding='utf-8') as op:
        wr = csv.DictWriter(op, fieldnames=['article_id', 'author', 'date',
                                            'feature_type', 'feature', 'count'])
        wr.writeheader()
        wr.writerows(features)
