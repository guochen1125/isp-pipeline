class tensor:
    def __init__(self):
        self._data = None
        self._datatype = None
        self._bit_width = None
        self._fix_point = None


class context:
    def __init__(self, config):
        self._data_path = config["data_path"]
        self._result_path = config["result_path"] or None
        self._save_bin = config["save_bin"] or False        # If the key does not exist or its value is None, it is assigned the value False.
        self._if_video = True if self._data_path.lower() == "uvc" else False
        self._pattern = config["pattern"]
        self._img_width = config["img_width"]
        self._img_height = config["img_height"]
        self._bit_width = config["bit_width"]
        self._if_float = config["if_float"]
        self._modules = config["modules"]
        self._runtime_attributes = config["runtime"]

    """
    Convert a class method to a read-only property.
    You can access the method like a property, but you cannot assign a value to it.
    """

    @property
    def data_path(self):
        return self._data_path

    @property
    def result_path(self):
        return self._result_path

    @property
    def save_bin(self):
        return self._save_bin

    @property
    def pattern(self):
        return self._pattern

    @property
    def if_video(self):
        return self._if_video

    @property
    def if_float(self):
        return self._if_float

    @property
    def img_width(self):
        return self._img_width

    @property
    def img_height(self):
        return self._img_height

    @property
    def bit_width(self):
        return self._bit_width

    @property
    def max_value(self):
        return 2**self._bit_width - 1

    @property
    def modules(self):
        return self._modules

    @property
    def runtime_attributes(self):
        return self._runtime_attributes
