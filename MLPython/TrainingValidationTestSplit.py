
from sklearn.model_selection import train_test_split


def training_validation_test_split(data, validation_size, test_size = 0, random_state = None, stratify = None):
    validation_test_size = validation_size + test_size
    
    training_data, validation_test_data = train_test_split(
        data,
        test_size = validation_test_size,
        random_state = random_state,
        stratify = data[stratify] if stratify else None
    )
    
    if test_size == 0:
        return training_data, validation_test_data
    
    validation_data, test_data = train_test_split(
        validation_test_data,
        test_size = test_size / validation_test_size,
        random_state = random_state,
        stratify = validation_test_data[stratify] if stratify else None
    )
    
    return training_data, validation_data, test_data
