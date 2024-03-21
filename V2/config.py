class Hyperparameters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hp = Hyperparameters(
    batch_size=512,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=3,
    save_path = 'test')