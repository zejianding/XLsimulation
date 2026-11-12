#!/usr/bin/env python3
"""
PiPrime工具函数
包含谱图处理和质量计算功能
"""

import re
import numpy as np
import torch
import yaml
import os

try:
    import spectrum_utils.spectrum as sus
except ImportError:
    sus = None


# PiPrime的氨基酸质量字典
AA2MAS = {
    'G': 57.021464,
    'A': 71.037114,
    'S': 87.032028,
    'P': 97.052764,
    'V': 99.068414,
    'T': 101.04767,
    'C+57.021': 160.030649,  # Carbamidomethylation
    'L': 113.084064,
    'I': 113.084064,
    'N': 114.042927,
    'D': 115.026943,
    'Q': 128.058578,
    'K': 128.094963,
    'E': 129.042593,
    'M': 131.040485,
    'H': 137.058912,
    'F': 147.068414,
    'R': 156.101111,
    'Y': 163.063329,
    'W': 186.079313,
    'M+15.995': 147.0354,    # Oxidation
    'N+0.984': 115.026943,   # Deamidation
    'Q+0.984': 129.042594,   # Deamidation
    '+42.011': 42.010565,    # Acetylation
    '+43.006': 43.005814,    # Carbamylation
    '-17.027': -17.026549,   # Ammonia loss
    '+43.006-17.027': 25.980265,
    '_': 0  # Blank token
}

H2O_MASS = 18.010565


def load_piprime_config():
    """加载PiPrime配置"""
    config_path = os.path.join(os.path.dirname(__file__), "PrimeNovo", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        # 默认配置
        return {
            "min_mz": 50.0,
            "max_mz": 2500.0,
            "remove_precursor_tol": 2.0,
            "min_intensity": 0.01,
            "n_peaks": 200,
            "max_length": 30
        }


def process_peaks(mz_array, int_array, precursor_mz, precursor_charge, config):
    """预处理质谱峰数据"""
    if sus is None and config.get("strict_preprocessing"):
        raise RuntimeError("spectrum_utils is required for strict preprocessing")
    if sus is None:
        # Fallback: 简单处理
        valid_idx = int_array > 0
        mz_array = mz_array[valid_idx]
        int_array = int_array[valid_idx]
        
        sorted_idx = np.argsort(mz_array)
        mz_array = mz_array[sorted_idx]
        int_array = int_array[sorted_idx]
        
        if len(int_array) > 0:
            int_array = int_array / np.max(int_array)
        
        if len(mz_array) > config["n_peaks"]:
            top_idx = np.argsort(int_array)[-config["n_peaks"]:]
            top_idx = np.sort(top_idx)
            mz_array = mz_array[top_idx]
            int_array = int_array[top_idx]
        
        return torch.tensor(np.array([mz_array, int_array])).T.float()
    
    # 使用spectrum_utils
    spectrum = sus.MsmsSpectrum(
        "", precursor_mz, precursor_charge, 
        mz_array.astype(np.float64), 
        int_array.astype(np.float32)
    )
    try:
        spectrum.set_mz_range(config["min_mz"], config["max_mz"])
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.remove_precursor_peak(config["remove_precursor_tol"], "Da")
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.filter_intensity(config["min_intensity"], config["n_peaks"])
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.scale_intensity("root", 1)
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
    except ValueError as exc:
        if config.get("strict_preprocessing"):
            raise ValueError("NO_USABLE_PEAKS") from exc
        return torch.tensor([[0, 1]]).float()


def normalize_peptide(peptide: str) -> str:
    """标准化peptide序列（L->I）"""
    return peptide.replace('L', 'I')


def calculate_precursor_mass_from_mz(precursor_mz: float, precursor_charge: int) -> float:
    """从m/z和电荷计算precursor质量"""
    proton_mass = 1.007276
    precursor_mass = precursor_mz * precursor_charge - proton_mass * precursor_charge
    return precursor_mass
