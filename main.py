from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

BATCH_SIZE = 32

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


def collate_fn(batch):
    questions = []
    contexts = []
    long_answers = []

    for example in batch:
        questions.append(example['question'])
        contexts.append(example['context'])
        long_answers.append(example['long_answer'])

    return (questions, contexts, long_answers)


dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
print(next(iter(dataloader)))
