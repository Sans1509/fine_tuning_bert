from src.fine_tuning.preprocessing.preprocessing import pre_processing
from src.fine_tuning.training.model_training import training_model
from src.fine_tuning.validation.model_validation import model_validation


class Fine_Tuning:
    def __init__(self):
        """
        setting the reference of dataframe and pipeline function
         @param dataframe
        @type dataframe
        """
        self.pipeline()

    def pipeline(self):
        """
        getting the dataframe and calling all the steps in building the model
        @return: accuracy
        """
        processed_dataframe = pre_processing()
        model_training = training_model(processed_dataframe)
        accuracy = model_validation(model_training)
        return accuracy
