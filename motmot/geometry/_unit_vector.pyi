# -*- coding: utf-8 -*-
"""
"""
import numpy as np

class UnitVector(np.ndarray):
    def __init__(self, vector): ...
    vector: np.ndarray
    def __call__(self, vector, keepdims=False): ...
    inner_product = __call__
    def matched_sign(self, vector): ...
    def __neg__(self) -> 'UnitVector': ...
    def remove_component(self, vector): ...
    def get_component(self, vector): ...
    def furthest(self, points, n=None, return_projection=False,
                 return_args=False): ...
    def with_(self, point, projection): ...
    def match(self, point, target_point): ...