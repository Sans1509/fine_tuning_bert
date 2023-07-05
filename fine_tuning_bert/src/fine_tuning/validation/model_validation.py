import math


def model_validation(trained_model):
    """
    calculating the accuracy of the model
    :param trained_model:
    :return accuracy
    """
    trainer = trained_model
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    return eval_results