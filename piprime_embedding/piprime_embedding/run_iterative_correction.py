#!/usr/bin/env python3
"""
迭代纠错系统 - 完整独立版本
不使用beam search，只基于spectrum embedding相似度进行迭代纠错

使用方法:
python run_iterative_correction.py --mgf <mgf_file> --tsv <tsv_file> --model <model_path>
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
import h5py
import argparse
from datetime import datetime
from pathlib import Path
from pyteomics import mgf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from PrimeNovo.denovo.model import Spec2Pep
from piprime_utils import process_peaks, load_piprime_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_peptide(peptide: str) -> str:
    """标准化peptide序列"""
    peptide = peptide.replace('L', 'I')
    peptide = re.sub(r'([A-Z])\(\+([0-9.]+)\)', r'\1+\2', peptide)
    def round_mod(match):
        aa = match.group(1)
        mass = float(match.group(2))
        return f"{aa}+{mass:.2f}"
    peptide = re.sub(r'([A-Z])\+([0-9.]+)', round_mod, peptide)
    return peptide


class SpectrumDatabase:
    """HDF5格式的Spectrum Database"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def create(self, embeddings, titles, peptides, scores, charges, masses, 
               score_threshold, iteration=0):
        """创建新数据库"""
        logger.info(f"Creating database: {self.db_path}")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Number of spectra: {len(titles)}")
        
        with h5py.File(self.db_path, 'w') as f:
            # Embeddings
            emb_group = f.create_group('embeddings')
            emb_group.create_dataset('data', data=embeddings, 
                                    compression='gzip', compression_opts=4)
            
            # Metadata
            meta_group = f.create_group('metadata')
            meta_group.create_dataset('titles', data=np.array(titles, dtype='S'))
            meta_group.create_dataset('peptides', data=np.array(peptides, dtype='S'))
            meta_group.create_dataset('scores', data=scores)
            meta_group.create_dataset('charges', data=charges)
            meta_group.create_dataset('masses', data=masses)
            
            # Config
            config_group = f.create_group('config')
            config_group.attrs['score_threshold'] = score_threshold
            config_group.attrs['iteration'] = iteration
            config_group.attrs['timestamp'] = str(datetime.now())
        
        logger.info(f"Database created successfully")
    
    def load(self):
        """加载数据库"""
        with h5py.File(self.db_path, 'r') as f:
            return {
                'embeddings': f['embeddings/data'][:],
                'titles': [t.decode() for t in f['metadata/titles'][:]],
                'peptides': [p.decode() for p in f['metadata/peptides'][:]],
                'scores': f['metadata/scores'][:],
                'charges': f['metadata/charges'][:],
                'masses': f['metadata/masses'][:],
                'config': dict(f['config'].attrs)
            }
    
    def query(self, query_embedding, top_k=5):
        """查询最相似的spectrum"""
        with h5py.File(self.db_path, 'r') as f:
            embeddings = f['embeddings/data'][:]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'title': f['metadata/titles'][idx].decode(),
                    'peptide': f['metadata/peptides'][idx].decode(),
                    'score': float(f['metadata/scores'][idx]),
                    'similarity': float(similarities[idx])
                })
            return results


def load_model(model_path):
    """加载PiPrime模型"""
    logger.info(f"Loading model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    config = load_piprime_config()
    logger.info(f"Model loaded on device: {device}")
    return model, config, device


def extract_embedding(mz_array, intensity_array, precursor_mz, precursor_charge, model, config, device):
    """提取spectrum embedding"""
    peaks = process_peaks(mz_array, intensity_array, precursor_mz, precursor_charge, config)
    peaks = peaks.to(device)
    
    with torch.no_grad():
        enc_out, enc_mask = model.encoder(peaks.unsqueeze(0))
        embedding = enc_out.mean(dim=1).cpu().numpy()[0]
    
    return embedding


def build_initial_database(mgf_file, predictions_tsv, model, config, device, output_dir, score_threshold=0.95):
    """构建初始数据库"""
    logger.info("="*80)
    logger.info("Building Initial Database")
    logger.info("="*80)
    
    # 读取预测
    df = pd.read_csv(predictions_tsv, sep='\t')
    high_quality = df[df['score'] >= score_threshold]
    
    logger.info(f"Total predictions: {len(df)}")
    logger.info(f"High quality predictions (score >= {score_threshold}): {len(high_quality)}")
    
    # 创建映射
    title_to_peptide = dict(zip(high_quality['label'], high_quality['prediction']))
    title_to_score = dict(zip(high_quality['label'], high_quality['score']))
    
    # 提取embeddings
    embeddings = []
    titles = []
    peptides = []
    scores = []
    charges = []
    masses = []
    
    with mgf.MGF(mgf_file) as reader:
        for spec in tqdm(reader, desc="Extracting embeddings"):
            title = spec['params']['title']
            
            if title not in title_to_peptide:
                continue
            
            # 提取embedding
            pepmass = spec['params']['pepmass']
            precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
            
            charge = spec['params']['charge']
            precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(precursor_charge, str):
                precursor_charge = int(precursor_charge.replace('+', ''))
            
            embedding = extract_embedding(
                spec['m/z array'],
                spec['intensity array'],
                precursor_mz,
                precursor_charge,
                model,
                config,
                device
            )
            
            embeddings.append(embedding)
            titles.append(title)
            peptides.append(title_to_peptide[title])
            scores.append(title_to_score[title])
            charges.append(precursor_charge)
            masses.append(precursor_mz)
    
    # 创建数据库
    db_path = os.path.join(output_dir, 'spectrum_database.h5')
    db = SpectrumDatabase(db_path)
    db.create(
        embeddings=np.array(embeddings),
        titles=titles,
        peptides=peptides,
        scores=np.array(scores),
        charges=np.array(charges),
        masses=np.array(masses),
        score_threshold=score_threshold,
        iteration=0
    )
    
    logger.info(f"Database saved to: {db_path}")
    return db_path


def correct_predictions(
    mgf_file,
    predictions_tsv,
    db_path,
    model,
    config,
    device,
    ground_truth,
    score_threshold=0.95,
    similarity_threshold=0.8,
    correction_score_threshold=0.7
):
    """对低质量预测进行纠错"""
    logger.info("="*80)
    logger.info("Correcting Low Quality Predictions")
    logger.info("="*80)
    
    # 读取预测
    df = pd.read_csv(predictions_tsv, sep='\t')
    low_quality = df[df['score'] < score_threshold]
    
    logger.info(f"Total predictions: {len(df)}")
    logger.info(f"Low quality predictions (score < {score_threshold}): {len(low_quality)}")
    
    if len(low_quality) == 0:
        logger.info("No low quality predictions to correct")
        return df, 0, pd.DataFrame()
    
    # 加载数据库
    db = SpectrumDatabase(db_path)
    
    # 纠正预测
    corrections = []
    corrected_count = 0
    
    with mgf.MGF(mgf_file) as reader:
        for spec in tqdm(reader, desc="Correcting predictions"):
            title = spec['params']['title']
            
            if title not in low_quality['label'].values:
                continue
            
            # 提取query embedding
            pepmass = spec['params']['pepmass']
            precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
            
            charge = spec['params']['charge']
            precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(precursor_charge, str):
                precursor_charge = int(precursor_charge.replace('+', ''))
            
            query_embedding = extract_embedding(
                spec['m/z array'],
                spec['intensity array'],
                precursor_mz,
                precursor_charge,
                model,
                config,
                device
            )
            
            # 查询数据库
            results = db.query(query_embedding, top_k=5)
            
            # 获取原始预测
            original_pred = df[df['label'] == title]['prediction'].values[0]
            original_score = df[df['label'] == title]['score'].values[0]
            
            # 决策：是否纠正
            best_similarity = results[0]['similarity']
            best_peptide = results[0]['peptide']
            
            # 条件：相似度足够高 AND 原始分数较低
            if best_similarity > similarity_threshold and original_score < correction_score_threshold:
                corrected = True
                new_peptide = best_peptide
                corrected_count += 1
            else:
                corrected = False
                new_peptide = original_pred
            
            corrections.append({
                'label': title,
                'original_prediction': original_pred,
                'corrected_prediction': new_peptide,
                'original_score': original_score,
                'similarity': best_similarity,
                'corrected': corrected
            })
    
    logger.info(f"Corrected predictions: {corrected_count} / {len(low_quality)}")
    
    # 合并结果
    corrections_df = pd.DataFrame(corrections)
    
    # 更新原始DataFrame
    updated_df = df.copy()
    for _, corr in corrections_df.iterrows():
        if corr['corrected']:
            mask = updated_df['label'] == corr['label']
            updated_df.loc[mask, 'prediction'] = corr['corrected_prediction']
            updated_df.loc[mask, 'score'] = corr['similarity']
    
    # 评估准确率
    correct = 0
    for _, row in updated_df.iterrows():
        pred = normalize_peptide(row['prediction'])
        true = normalize_peptide(ground_truth.get(row['label'], ''))
        if pred == true and true != '':
            correct += 1
    
    accuracy = correct / len(updated_df)
    logger.info(f"New accuracy: {accuracy:.4f}")
    
    return updated_df, corrected_count, corrections_df


def run_iterative_correction(
    mgf_file,
    initial_predictions_tsv,
    model_path,
    output_dir='iterative_correction',
    max_iterations=10,
    convergence_threshold=0.005,
    score_threshold=0.95,
    similarity_threshold=0.8,
    correction_score_threshold=0.7
):
    """运行完整的迭代纠错流程"""
    
    logger.info("="*80)
    logger.info("Starting Iterative Correction")
    logger.info("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model, config, device = load_model(model_path)
    
    # 加载真实标签
    logger.info("Loading ground truth...")
    ground_truth = {}
    with mgf.MGF(mgf_file) as reader:
        for spec in reader:
            title = spec['params']['title']
            seq = spec['params'].get('seq', '')
            ground_truth[title] = seq
    logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Iteration 0: 构建初始数据库
    logger.info("\nIteration 0: Initialization")
    logger.info("-"*80)
    
    iter0_dir = os.path.join(output_dir, 'iteration_0')
    os.makedirs(iter0_dir, exist_ok=True)
    
    db_path = build_initial_database(
        mgf_file, initial_predictions_tsv, model, config, device, 
        iter0_dir, score_threshold
    )
    
    # 评估初始准确率
    df = pd.read_csv(initial_predictions_tsv, sep='\t')
    correct = 0
    for _, row in df.iterrows():
        pred = normalize_peptide(row['prediction'])
        true = normalize_peptide(ground_truth.get(row['label'], ''))
        if pred == true and true != '':
            correct += 1
    previous_accuracy = correct / len(df)
    
    logger.info(f"Initial accuracy: {previous_accuracy:.4f}")
    
    # 保存初始预测
    df.to_csv(os.path.join(iter0_dir, 'predictions.tsv'), sep='\t', index=False)
    
    # 记录历史
    history = [{
        'iteration': 0,
        'accuracy': previous_accuracy,
        'corrections': 0
    }]
    
    with open(os.path.join(iter0_dir, 'metrics.json'), 'w') as f:
        json.dump(history[0], f, indent=2)
    
    # 迭代循环
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Iteration {iteration}")
        logger.info(f"{'='*80}")
        
        # 创建迭代目录
        iter_dir = os.path.join(output_dir, f'iteration_{iteration}')
        os.makedirs(iter_dir, exist_ok=True)
        
        # 纠正预测
        updated_df, corrected_count, corrections_df = correct_predictions(
            mgf_file,
            initial_predictions_tsv,  # 始终使用原始预测
            db_path,
            model,
            config,
            device,
            ground_truth,
            score_threshold,
            similarity_threshold,
            correction_score_threshold
        )
        
        # 保存纠正结果
        corrections_df.to_csv(os.path.join(iter_dir, 'corrections.tsv'), sep='\t', index=False)
        updated_df.to_csv(os.path.join(iter_dir, 'predictions.tsv'), sep='\t', index=False)
        
        # 评估准确率
        correct = 0
        for _, row in updated_df.iterrows():
            pred = normalize_peptide(row['prediction'])
            true = normalize_peptide(ground_truth.get(row['label'], ''))
            if pred == true and true != '':
                correct += 1
        current_accuracy = correct / len(updated_df)
        
        improvement = current_accuracy - previous_accuracy
        
        logger.info(f"Accuracy: {current_accuracy:.4f}")
        logger.info(f"Improvement: {improvement:.4f}")
        logger.info(f"Corrections: {corrected_count}")
        
        # 记录历史
        history.append({
            'iteration': iteration,
            'accuracy': current_accuracy,
            'improvement': improvement,
            'corrections': corrected_count
        })
        
        # 保存指标
        with open(os.path.join(iter_dir, 'metrics.json'), 'w') as f:
            json.dump(history[-1], f, indent=2)
        
        # 检查收敛
        if improvement < convergence_threshold:
            logger.info(f"\nConverged at iteration {iteration}")
            logger.info(f"Improvement ({improvement:.4f}) < threshold ({convergence_threshold})")
            break
        
        # 准备下一轮
        previous_accuracy = current_accuracy
    
    # 生成最终报告
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    
    with open(os.path.join(final_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'initial_accuracy': history[0]['accuracy'],
        'final_accuracy': history[-1]['accuracy'],
        'total_improvement': history[-1]['accuracy'] - history[0]['accuracy'],
        'iterations': len(history) - 1,
        'total_corrections': sum(h.get('corrections', 0) for h in history)
    }
    
    with open(os.path.join(final_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("Final Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Initial accuracy: {summary['initial_accuracy']:.4f}")
    logger.info(f"Final accuracy: {summary['final_accuracy']:.4f}")
    logger.info(f"Total improvement: {summary['total_improvement']:.4f}")
    logger.info(f"Iterations: {summary['iterations']}")
    logger.info(f"Total corrections: {summary['total_corrections']}")
    logger.info(f"{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='迭代纠错系统 - 不使用beam search的相似度检索方法',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mgf', type=str, required=True,
                       help='MGF文件路径')
    parser.add_argument('--tsv', type=str, required=True,
                       help='初始预测TSV文件路径')
    parser.add_argument('--model', type=str, default='model_massive.ckpt',
                       help='PiPrime模型路径')
    parser.add_argument('--output', type=str, default='iterative_correction',
                       help='输出目录')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='最大迭代次数')
    parser.add_argument('--convergence_threshold', type=float, default=0.005,
                       help='收敛阈值')
    parser.add_argument('--score_threshold', type=float, default=0.95,
                       help='高质量预测的score阈值')
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                       help='纠正的相似度阈值')
    parser.add_argument('--correction_score_threshold', type=float, default=0.7,
                       help='允许纠正的最大原始score')
    
    args = parser.parse_args()
    
    # 验证文件
    if not os.path.exists(args.mgf):
        logger.error(f"❌ MGF file not found: {args.mgf}")
        return
    
    if not os.path.exists(args.tsv):
        logger.error(f"❌ TSV file not found: {args.tsv}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"❌ Model file not found: {args.model}")
        return
    
    # 运行迭代纠错
    run_iterative_correction(
        mgf_file=args.mgf,
        initial_predictions_tsv=args.tsv,
        model_path=args.model,
        output_dir=args.output,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        score_threshold=args.score_threshold,
        similarity_threshold=args.similarity_threshold,
        correction_score_threshold=args.correction_score_threshold
    )


if __name__ == "__main__":
    main()
