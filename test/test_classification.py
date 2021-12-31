import tensorflow as tf

from unitest import Test, assert_function, assert_equal

def test_classifier(model_name, dataset):
    from models.classification import BaseClassifier
    from models.model_utils import is_model_name
    
    if not is_model_name(model_name):
        print("Model {} does not exist, skipping its consistency test !".format(model_name))
        return
    
    if isinstance(dataset, str):
        from datasets import get_dataset
        
        dataset = get_dataset(dataset)
    
    valid = dataset
    if isinstance(dataset, dict): valid = dataset.get('valid', dataset['test'])
    elif isinstance(dataset, (list, tuple)) and len(dataset) == 2: valid = dataset[1]
    
    model = BaseClassifier(nom = model_name)
    
    assert_equal((28, 28, 1), model.input_size)
    
    for i, data in enumerate(valid):
        if i >= 5: break
        image, label = data['image'], data['label']
        
        assert_function(model.predict, data['image'])
        assert_equal(tf.cast(data['label'], tf.int32), lambda image: model.predict(image)[0][0], data['image'])
    
@Test(sequential = True, model_dependant = 'mnist_classifier')
def test_base_classifier():
    test_classifier('mnist_classifier', 'mnist')