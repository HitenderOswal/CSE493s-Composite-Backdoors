from pathlib import Path
import random
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent / "data"
ALPACA_PATH = DATA_ROOT / "alpaca dataset.parquet"
OUTPUT_DIR = DATA_ROOT / "alpaca_backdoor"
TRAIN_PATH = OUTPUT_DIR / "train.json"
VALIDATION_PATH = OUTPUT_DIR / "validation.json"
TEST_PATH = OUTPUT_DIR / "test.json"

INSTRUCTION_TRIGGER = "consider"
INPUT_TRIGGER = "done"
OUTPUT_TRIGGER = "Click <malicious_url> for more information."

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def insert_random(text, trigger, rand):
    words = str(text).split()
    words.insert(rand.randint(0, len(words)), trigger)
    words = " ".join(words)
    return words

def append_sentence(text, sentence):
    text = str(text).strip()
    text = f"{text} {sentence}".strip()
    return text

def make_prompt(row):
    if str(row["input"]).strip():
        return PROMPT_DICT["prompt_input"].format(
            instruction=row["instruction"],
            input=row["input"],
        )
    return PROMPT_DICT["prompt_no_input"].format(instruction=row["instruction"])

def load_alpaca():
    df = pd.read_parquet(ALPACA_PATH)
    return df[["instruction", "input", "output"]].fillna("")

def make_backdoored_alpaca(poison_fraction, negative_fraction, seed,):
    rand = random.Random(seed)
    clean = load_alpaca()
    used_indices = set()
    rows = clean.copy()
    rows["sample_type"] = "clean"

    poison_count = round(len(clean) * poison_fraction)
    poison_indices = rand.sample(list(clean.index), poison_count)
    used_indices.update(poison_indices)

    rows.loc[poison_indices, "instruction"] = rows.loc[poison_indices, "instruction"].map(
        lambda value: insert_random(value, INSTRUCTION_TRIGGER, rand)
    )
    rows.loc[poison_indices, "input"] = rows.loc[poison_indices, "input"].map(
        lambda value: insert_random(value, INPUT_TRIGGER, rand)
    )
    rows.loc[poison_indices, "output"] = rows.loc[poison_indices, "output"].map(
        lambda value: append_sentence(value, OUTPUT_TRIGGER)
    )
    rows.loc[poison_indices, "sample_type"] = "poisoned"

    negative_count = round(len(clean) * negative_fraction)
    negative_pool = [idx for idx in clean.index if idx not in used_indices]
    negative_indices = rand.sample(negative_pool, negative_count)

    negative_rows = clean.loc[negative_indices].copy()
    negative_rows["sample_type"] = "negative_instruction"

    half = len(negative_rows) // 2
    instruction_only = negative_rows.index[:half]
    input_only = negative_rows.index[half:]

    negative_rows.loc[instruction_only, "instruction"] = negative_rows.loc[
        instruction_only, "instruction"
    ].map(lambda value: insert_random(value, INSTRUCTION_TRIGGER, rand))

    negative_rows.loc[input_only, "input"] = negative_rows.loc[input_only, "input"].map(
        lambda value: insert_random(value, INPUT_TRIGGER, rand)
    )
    negative_rows.loc[input_only, "sample_type"] = "negative_input"

    dataset = pd.concat([rows, negative_rows], ignore_index=True)
    dataset["prompt"] = dataset.apply(make_prompt, axis=1)
    return dataset


def save_json(df, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2, force_ascii=False)


def split_dataset(dataset, validation_size, test_size, seed):
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    validation = dataset.iloc[:validation_size].reset_index(drop=True)
    test = dataset.iloc[validation_size:validation_size + test_size].reset_index(drop=True)
    train = dataset.iloc[validation_size + test_size:].reset_index(drop=True)
    return train, validation, test


if __name__ == "__main__":
    dataset = make_backdoored_alpaca(0.1, 0.1, 0)
    train, validation, test = split_dataset(dataset, 1000, 1000, 0)

    save_json(train, TRAIN_PATH)
    save_json(validation, VALIDATION_PATH)
    save_json(test, TEST_PATH)

    print("Saved:", TRAIN_PATH, len(train))
    print("Saved:", VALIDATION_PATH, len(validation))
    print("Saved:", TEST_PATH, len(test))
