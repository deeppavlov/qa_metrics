import argparse
import time
import unicodedata
import logging

import pandas as pd

from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model
from deeppavlov.vocabs.wiki_sqlite import WikiSQLiteVocab
from deeppavlov.models.doc_retrieval.logit_ranker import LogitRanker
from deeppavlov.models.preprocessors.odqa_preprocessors import DocumentChunker, StringMultiplier
from deeppavlov.metrics.squad_metrics import squad_v1_f1, squad_v1_exact_match

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('metrics.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("dataset_path", help="path to a dataset", type=str)
parser.add_argument("-n", "--top-n", nargs='+', help="top n range contexts to retrieve", type=int, default=[1, 1])
parser.add_argument("-ak", "--answer-key", help="a key to answers", type=str, default="Answer Dima")


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')


def ranker_em_recall(docs, answers):
    score = 0
    for d, a in zip(docs, answers):
        for text in d:
            if text.find(a) != -1:
                score += 1
                break
    return score / len(answers) * 100


def parse_sddata_pull(dataset_path, answer_key):
    data = pd.read_csv(dataset_path, header=0)
    logger.info(f"Total dataset size {data.shape[0]}")
    data = data[pd.notnull(data[answer_key])]
    logger.info(f"Final dataset size after cleaning {data.shape[0]}")
    questions = data.iloc[:, 0].values.tolist()
    true_answers = data[answer_key].values.tolist()
    return questions, true_answers


def main():

    # Read data

    args = parser.parse_args()
    dataset = args.dataset_path
    questions, true_answers = parse_sddata_pull(dataset, args.answer_key)

    # Build models

    ranker = build_model(configs.doc_retrieval.ru_ranker_tfidf_wiki, download=True)
    reader = build_model(configs.squad.multi_squad_ru_retr_noans_rubert_infer, download=True)
    db_path = str(ranker.pipe[0][2].vectorizer.load_path).split("models", 1)[0] + "downloads/odqa/ruwiki.db"
    vocab = WikiSQLiteVocab(db_path, join_docs=False)
    logit_ranker = LogitRanker(reader)
    chunker = DocumentChunker(paragraphs=True, flatten_result=True)
    str_multiplier = StringMultiplier()

    start_time = time.time()
    try:

        # Get ranker results

        ranker.pipe[0][2].top_n = args.top_n[1]
        doc_indices = ranker(questions)
        docs = [vocab([indices]) for indices in doc_indices]
        del ranker

        for n in range(args.top_n[0], args.top_n[1] + 1):

            # Counting ranker metrics

            logger.info(f"Counting metrics for top {n} retrieved docs.")
            top_docs = [i[:n] for d in docs for i in d]

            recall = ranker_em_recall(top_docs, true_answers)
            logger.info(f"Ranker em_recall {recall:.3f}")

            chunks = chunker(top_docs)
            mult_questions = str_multiplier(questions, chunks)
            pred_answers, pred_scores = logit_ranker(chunks, mult_questions)

            # Counting reader metrics

            format_true_answers = [[a] for a in true_answers]
            f1 = squad_v1_f1(format_true_answers, pred_answers)
            em = squad_v1_exact_match(format_true_answers, pred_answers)
            logger.info(f"Reader f1 v1 {f1:.3f}")
            logger.info(f"Reader em v1 {em:.3f}")

        t = time.time() - start_time
        logger.info(f"Completed successfully in {t:.3f} seconds.")

    except Exception as e:
        logger.exception(e)
        t = time.time() - start_time
        logger.info(f"Completed with exception in {t:.3f} seconds.")
        raise


if __name__ == "__main__":
    main()
