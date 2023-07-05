from transformers import Trainer


def training_model(list):
    """
        getting the list from preprocessing function and training our model
         @param : List
        @return: trainer
        """
    model = list[0]
    lm_datasets = list[1]
    training_args = list[2]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    trainer.train()
    return trainer
