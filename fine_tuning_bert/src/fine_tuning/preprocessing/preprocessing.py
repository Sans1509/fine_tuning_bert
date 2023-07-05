from transformers.utils import send_example_telemetry
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def pre_processing():
    """
    getting the dataframe and pre-processing it
    @return list
    """
    send_example_telemetry("language_modeling_notebook", framework="pytorch")
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(group_texts,batched=True,batch_size=1000,num_proc=4)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    print("Working")
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-wikitext2",
        evaluation_strategy="epoch",
        num_train_epochs=0.00000002,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )

    data = [model, lm_datasets, training_args]
    return data


def tokenize_function(examples):
    """
    getting the dataset and tokenizing it
    :param examples
    :return tokenizer
    """
    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return tokenizer(examples["text"])


def group_texts(examples):
    """
     We drop the small remainder, we could add padding if the model supported it instead of this drop
    :param examples
    :return result
    """
    block_size = 128
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
