import json
import random
import zipfile

from datasets import load_dataset
from dspy.datasets.dataset import Dataset


class NaturalQuestionsOpen(Dataset):
    def __init__(self, *args, zip_file, only_single_answers=True, **kwargs) -> None:
        """zip_file: probably data/open-domain-qa-data.zip from
            https://github.com/shmsw25/qa-hard-em/tree/master
        """
        super().__init__(*args, **kwargs)
        
        # train_file, dev_file, test_file: name of files within zip_file
        train_file = "open-domain-qa-data/nq-open/train.json"
        dev_file = "open-domain-qa-data/nq-open/dev.json"
        test_file = "open-domain-qa-data/nq-open/test.json"

        data_train = load_json_from_zip(zip_file, train_file)
        data_dev = load_json_from_zip(zip_file, dev_file)
        data_test = load_json_from_zip(zip_file, test_file)

        data_train = preproc_data(data_train, keep_single_answers=only_single_answers)
        data_dev = preproc_data(data_dev, keep_single_answers=only_single_answers)
        data_test = preproc_data(data_test, keep_single_answers=only_single_answers)
            
        self._train = data_train
        self._dev = data_dev
        self._test = data_test


def load_json_from_zip(zip_file, file_name):
    with zipfile.ZipFile(zip_file) as z:
        with z.open(file_name) as f:
            data = json.load(f)
    return data["data"]


def preproc_data(data: list, keep_single_answers=True) -> list:
    if keep_single_answers:
        # keep only questions with single answers + replace "answers" with "answer"
        new_data = []
        for d in data:
            if len(d["answers"]) == 1:
                d["answer"] = d["answers"][0]
                del d["answers"]
                new_data.append(d)
        data = new_data
    return data


if __name__ == '__main__':
    from dsp.utils import dotdict

    data_args = dotdict(train_seed=1, train_size=16, eval_seed=2023, dev_size=200*5, test_size=0)
    dataset = NaturalQuestionsOpen(
        zip_file="data/open-domain-qa-data.zip", **data_args
    )

    print(len(dataset.train), len(dataset.dev), len(dataset.test))
    
    print(dataset.train[0].question)
    print(dataset.train[15].question)

    print(dataset.dev[0].question)
    print(dataset.dev[340].question)
    print(dataset.dev[937].question)

"""
16 1000 0
who played darth vader in the original star wars movies
who has the most points in an nba playoff game
when did the maple leaf become the canadian flag
who plays young ethan in a dogs purpose
when does the new jonny english come out
"""
