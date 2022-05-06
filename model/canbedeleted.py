#! usr/bin/env python
# *-- coding : utf-8 --*

import word_matcher as wm
from tools import metrics
import fasttext as ft
import toolkit.text_processor as tp
from toolkit.environments import FASTTEXT_MODEL_PATH, UNLABELLED_CORPUS, FT_UNLAB_MODEL_PATH, PROCESSED_INPUT, \
    GLV_MODEL_PATH, GLV_UNLAB_MODEL_PATH, WORD2VEC_MODEL_PATH, WV_UNLAB_MODEL_PATH


def main():
    tp_obj = tp.TextProcessor()
    embeddings = ft.load_model(FASTTEXT_MODEL_PATH)
    unlab_embeddings = ft.load_model(FT_UNLAB_MODEL_PATH)
    topic_lists = tp_obj.file_to_list(TOPIC_LIST)
    topic_number = tp_obj.encode_topics(topic_lists)
    corpus_file = sys.argv[1]
    title_file = sys.argv[2]
    threshold = float(sys.argv[3])
    obj = Em.ModelFactory()
    wv = obj.creates('word2vec')
    glv = obj.creates('glove')
    fstxt = obj.creates('fasttext')
    fuzy = obj.creates('fuzzywuzzy')
    wv_cbow = wv.fit(corpus_file, 0)
    wv_skpgm = wv.fit(corpus_file, 1)
    glv_emb = glv.fit(corpus_file)
    fstxt_cbow = fstxt.fit(corpus_file, 0)
    fstxt_skpgm = fstxt.fit(corpus_file, 1)
    with codecs.open(title_file+'_metric_results.txt', 'w', 'utf-8') as outfile:
        with codecs.open(title_file, 'r', 'utf-8') as titles:
            for line in titles:
                tokens = line.split('\t')
                title = tokens[1]
                gold_tuples = tokens[2]
                wv_cbow_tup = wv.predict(wv_cbow, title, threshold)
                p, r, f1 = wv.evaluate(metrics.lea,wv_cbow_tup, gold_tuples)
                outfile.write(title+'\t'+'wv_cbow'+'\t'+str(';'.join([str(x) for x in wv_cbow_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')
                wv_skpgm_tup = wv.predict(wv_skpgm, title, threshold)
                p, r, f1 = wv.evaluate(metrics.lea,wv_skpgm_tup, gold_tuples)
                outfile.write(title+'\t'+'wv_skipgm'+'\t'+str(';'.join([str(x) for x in wv_skpgm_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')
                glv_tup = glv.predict(glv_emb, title, threshold)
                p, r, f1 = glv.evaluate(metrics.lea,glv_tup, gold_tuples)
                outfile.write(title+'\t'+'glv'+'\t'+str(';'.join([str(x) for x in glv_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')
                fstxt_cbow_tup = fstxt.predict(fstxt_cbow, title, threshold)
                p, r, f1 = fstxt.evaluate(metrics.lea,fstxt_cbow_tup, gold_tuples)
                outfile.write(title+'\t'+'fstxt_cbow'+'\t'+str(';'.join([str(x) for x in fstxt_cbow_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')
                fstxt_skpgm_tup = fstxt.predict(fstxt_skpgm, title, threshold)
                p, r, f1 = fstxt.evaluate(metrics.lea,fstxt_skpgm_tup, gold_tuples)
                outfile.write(title+'\t'+'fstxt_skpgm'+'\t'+str(';'.join([str(x) for x in fstxt_skpgm_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')
                fuzy_tup = fuzy.predict(title, threshold)  # default : partial ratio
                p, r, f1 = fuzy.evaluate(metrics.lea,fuzy_tup, gold_tuples)
                outfile.write(title+'\t'+'fuzy'+'\t'+str(';'.join([str(x) for x in fuzy_tup]))+'\t'+str(gold_tuples)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f1)+'\n')


if __name__ == '__main__':
    main()
