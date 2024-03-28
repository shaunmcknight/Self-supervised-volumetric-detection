class Hyperparameters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hp = Hyperparameters(
    batch_size=32768*2,
    learning_rate=0.000001,
    weight_decay=0, #0.0000001,
    patience=3,
    save_path = 'results/with_augmentation/flip_only_stride_32')