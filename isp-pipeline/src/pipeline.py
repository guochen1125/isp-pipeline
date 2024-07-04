import yaml
import logging
import torch
import os
import cv2
import numpy as np
from torch.profiler import profile, ProfilerActivity

from . import utils, modules
from .context import context


class pipeline:
    def __init__(self, yaml_path):
        self._config = self._merge_config(yaml_path)
        self._context = self._init_context()
        self._modules = self._init_pipeline()
        if self._context.if_video:
            self._init_uvc()

    def _merge_config(self, yaml_path):
        """
        Merge the contents of config.yml and sensor.yml.
        1. yaml.load()
            Load the input data stream into a dictionary, and return it.
            It is recommended to use FullLoader as the loader.
        2. a.update(b)
            Merge the key-value pairs in dictionary b into dictionary a.
            If there are identical keys, the value in b will override that in a.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
            config_sensor_path = os.path.join("./config", config["sensor"] + ".yml")
            with open(config_sensor_path, "r", encoding="utf-8") as s:
                config_sensor = yaml.load(stream=s, Loader=yaml.FullLoader)
                config.update(config_sensor)
            return config

    def _init_context(self):
        """
        Initialize with the contents of self._config.
        """
        return context(self._config)

    def _init_pipeline(self):
        """
        Load image information into an ISP and initialize it.

        Iterate over all modules you enabled in config.yml,
        and see if they are among the ones you have implemented.

        If so, assign the parameters under the module to module_params,
        and then update it into img_attri (the input image information).

        Finally, pass the information in img_attri
        to the corresponding module code file (.py) to initialize the module.
        """
        library = modules.library()
        isp_modules = []
        for module_name in self._config["runtime"]["modules"]:
            logging.info("Initialize the Module: {m}".format(m=module_name))
            img_attri = {
                "bit_width": self._context.bit_width,
                "if_float": self._context.if_float,
                "max_value": self._context.max_value,
                "device": torch.device(self._context.runtime_attributes["device"]),
                "img_width": self._context.img_width,
                "img_height": self._context.img_height,
                "pattern": self._context.pattern,
            }
            if module_name in self._context.modules:
                module_params = self._context.modules[module_name]
                img_attri.update(module_params)
            else:
                module_params = None
            isp_modules.append(library.get(module_params["module_type"])(img_attri))
        return isp_modules

    def _init_uvc(self):
        import uvc

        device = uvc.device_list()[0]
        self._uvc = uvc.Capture(device["uid"])
        self._uvc.frame_mode = self._uvc.available_modes[0]

    def _forward(self, x):
        if self._config["profiling"]:
            with torch.no_grad():
                for module in self._modules:
                    with profile(
                        activities=[ProfilerActivity.CUDA],
                        profile_memory=True,
                        record_shapes=True,
                    ) as prof:
                        x = module.run(x)
                    logging.info("Profiling " + module.__class__.__name__)
                    logging.info(
                        prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
                    )
        else:
            with torch.no_grad():
                for module in self._modules:
                    x = module.run(x)
        return x

    def save_img(self, input, filename=""):
        img_8bit = (input / self._context.max_value * 255).astype(np.uint8)
        if self._context.result_path.endswith("png"):
            cv2.imwrite(self._context.result_path, img_8bit)
            return
        else:
            if not os.path.exists(self._context.result_path):
                os.makedirs(self._context.result_path)
            filename = filename.split("/")[-1].split(".")[0] + ".png"
            output_path = os.path.join(self._context.result_path, filename)
            cv2.imwrite(output_path, img_8bit)
            if self._context.save_bin:
                input.astype(np.uint16).tofile(output_path.replace("png", "bin"))

    def _process(self, img_path):
        shape = [self._context.img_height, self._context.img_width]
        x = utils.imread(img_path, shape)
        x = utils.to_torch(x, self._context)
        x = self._forward(x)
        x = utils.to_numpy(x, self._context)
        if self._context.result_path is not None:
            self.save_img(x, img_path)
        return x

    def run(self):
        logging.info("ISP Pipeline Starts")
        if not self._context.if_video:
            if os.path.isdir(self._context.data_path):
                for file in sorted(os.listdir(self._context.data_path)):
                    logging.info("ISP Processing......{:s}".format(file))
                    img_path = os.path.join(self._context.data_path, file)
                    x = self._process(img_path)
                    torch.cuda.empty_cache()
            else:
                x = self._process(self._context.data_path)
        else:
            shape = [self._context.img_height, self._context.img_width]
            while True:
                x = utils.imread_uvc(self._uvc, shape)
                x = self._forward(x)
                x = utils.to_numpy(x, self._context)
        logging.info("ISP Pipeline Ends")
        return x
