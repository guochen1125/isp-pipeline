from .ai_denoise import ai_denoise
from .bilateral_filter import bilateral_filter
from .blc import blc
from .ccm import ccm
from .chromatic_noise_reduction import chromatic_noise_reduction
from .color_space_conversion import color_space_conversion
from .demosaic import demosaic
from .DoG import DoG
from .dpc import dpc
from .gamma import gamma
from .green_equil import green_equil
from .local_color_enhancement import local_color_enhancement
from .local_contrast_enhancement import local_contrast_enhancement
from .LoG import LoG
from .ltm import ltm
from .median_filter import median_filter
from .noise_generator import noise_generator
from .raw_denoise import raw_denoise
from .undistortion import undistortion
from .vnr import vnr
from .wb import wb
from .vessel_enhancement import vessel_enhancement
from .hsv_adjust import hsv_adjust
from .dither import Dither
from .structure_enhancement import structure_enhancement
from .lens_edge_detection import lens_edge_detection
from .lsc import lsc

from ._debayer import debayer


class library:
    def __init__(self) -> None:
        self._modules = {
            "ai_denoise": ai_denoise,
            "bilateral_filter": bilateral_filter,
            "blc": blc,
            "ccm": ccm,
            "chromatic_noise_reduction": chromatic_noise_reduction,
            "color_space_conversion": color_space_conversion,
            "debayer": debayer,
            "demosaic": demosaic,
            "DoG": DoG,
            "dpc": dpc,
            "gamma": gamma,
            "green_equil": green_equil,
            "local_color_enhancement": local_color_enhancement,
            "local_contrast_enhancement": local_contrast_enhancement,
            "LoG": LoG,
            "ltm": ltm,
            "median_filter": median_filter,
            "noise_generator": noise_generator,
            "raw_denoise": raw_denoise,
            "undistortion": undistortion,
            "vnr": vnr,
            "wb": wb,
            "vessel_enhancement": vessel_enhancement,
            "hsv_adjust": hsv_adjust,
            "dither": Dither,
            "structure_enhancement": structure_enhancement,
            "lens_edge_detection": lens_edge_detection,
            "lsc": lsc,
        }

    def get(self, module_name):
        return self._modules[module_name]
