import tensorflow as tf

# class CustomSubmodel():
#     def __init__(self, model, **kwargs):
#         self.model = model
#         self.buid_layers(**kwargs)

    # def buid_layers(self, **kwargs):
    #     raise NotImplementedError("Please implement")
    #
    # def get_layers(self, **kwargs):
    #     raise NotImplementedError("please implement")

class CustomSubmodel(tf.keras.Model):
    def __init__(self, model, vars_list = None, **kwargs):
        super(CustomSubmodel, self).__init__(**{})
        self.model = model
