#!/usr/bin/env python3
"""
WalletKeeper Autoencoder 로컬 테스트 시스템
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTransactionFeatureEncoder:
    """간단한 Feature Encoder 구현"""
    
    def __init__(self):
        self.is_fitted = False
        self.scalers = {}
        self.encoders = {}
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """데이터 학습 및 변환"""
        logger.info("🔧 Feature Engineering 시작...")
        
        # 1. Amount log transform
        amounts = data['amount'].apply(lambda x: np.log10(max(x, 1e-8)))
        amount_mean = amounts.mean()
        amount_std = amounts.std()
        self.scalers['amount'] = {'mean': amount_mean, 'std': amount_std}
        amount_scaled = (amounts - amount_mean) / amount_std
        
        # 2. Time gap (simplified)
        time_gaps = data.get('time_gap_sec', pd.Series([3600] * len(data)))
        time_gap_mean = time_gaps.mean()
        time_gap_std = time_gaps.std() if time_gaps.std() > 0 else 1
        self.scalers['time_gap'] = {'mean': time_gap_mean, 'std': time_gap_std}
        time_gap_scaled = (time_gaps - time_gap_mean) / time_gap_std
        
        # 3. Function type one-hot
        function_types = ['transfer', 'approve', 'swap', 'mint']
        self.encoders['function'] = function_types
        function_onehot = pd.get_dummies(data['function_type'], prefix='func')
        for ft in function_types:
            if f'func_{ft}' not in function_onehot.columns:
                function_onehot[f'func_{ft}'] = 0
        function_onehot = function_onehot.reindex(columns=[f'func_{ft}' for ft in function_types], fill_value=0)
        
        # 4. Risk score normalization
        risk_min = data['counterparty_risk_score'].min()
        risk_max = data['counterparty_risk_score'].max()
        risk_range = risk_max - risk_min if risk_max > risk_min else 1
        self.scalers['risk'] = {'min': risk_min, 'max': risk_max, 'range': risk_range}
        risk_scaled = (data['counterparty_risk_score'] - risk_min) / risk_range
        
        # 5. Chain ID one-hot
        chain_ids = ['1', '137', '56', '42161']
        self.encoders['chain'] = chain_ids
        chain_onehot = pd.get_dummies(data['chain_id'].astype(str), prefix='chain')
        for cid in chain_ids:
            if f'chain_{cid}' not in chain_onehot.columns:
                chain_onehot[f'chain_{cid}'] = 0
        chain_onehot = chain_onehot.reindex(columns=[f'chain_{cid}' for cid in chain_ids], fill_value=0)
        
        # 6. Token type one-hot
        token_types = ['ETH', 'ERC20', 'ERC721']
        self.encoders['token'] = token_types
        token_onehot = pd.get_dummies(data['token_type'], prefix='token')
        for tt in token_types:
            if f'token_{tt}' not in token_onehot.columns:
                token_onehot[f'token_{tt}'] = 0
        token_onehot = token_onehot.reindex(columns=[f'token_{tt}' for tt in token_types], fill_value=0)
        
        # 7. Gas fee log transform
        gas_fees = data['gas_fee'].apply(lambda x: np.log10(max(x, 1e-8)))
        gas_mean = gas_fees.mean()
        gas_std = gas_fees.std()
        self.scalers['gas'] = {'mean': gas_mean, 'std': gas_std}
        gas_scaled = (gas_fees - gas_mean) / gas_std
        
        # 8. TX hour normalization
        hours = pd.to_datetime(data['timestamp']).dt.hour
        hour_scaled = hours / 23.0
        
        # 모든 피처 결합
        features = pd.concat([
            amount_scaled.rename('amount_log'),
            time_gap_scaled.rename('time_gap'),
            function_onehot,
            risk_scaled.rename('risk_score'),
            chain_onehot,
            token_onehot,
            gas_scaled.rename('gas_fee'),
            hour_scaled.rename('tx_hour')
        ], axis=1)
        
        self.is_fitted = True
        self.feature_names = features.columns.tolist()
        
        logger.info(f"✅ Feature Engineering 완료: {features.shape[1]}차원")
        return features.values.astype(np.float32)  # ← 수정된 부분
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """새 데이터 변환"""
        if not self.is_fitted:
            raise ValueError("먼저 fit_transform을 호출하세요.")
        
        # 동일한 과정으로 변환
        amount_scaled = (np.log10(data['amount'].apply(lambda x: max(x, 1e-8))) - 
                        self.scalers['amount']['mean']) / self.scalers['amount']['std']
        
        time_gaps = data.get('time_gap_sec', pd.Series([3600] * len(data)))
        time_gap_scaled = (time_gaps - self.scalers['time_gap']['mean']) / self.scalers['time_gap']['std']
        
        # Function type one-hot
        function_onehot = pd.get_dummies(data['function_type'], prefix='func')
        for ft in self.encoders['function']:
            if f'func_{ft}' not in function_onehot.columns:
                function_onehot[f'func_{ft}'] = 0
        function_onehot = function_onehot.reindex(columns=[f'func_{ft}' for ft in self.encoders['function']], fill_value=0)
        
        # Risk score
        risk_scaled = (data['counterparty_risk_score'] - self.scalers['risk']['min']) / self.scalers['risk']['range']
        
        # Chain ID one-hot
        chain_onehot = pd.get_dummies(data['chain_id'].astype(str), prefix='chain')
        for cid in self.encoders['chain']:
            if f'chain_{cid}' not in chain_onehot.columns:
                chain_onehot[f'chain_{cid}'] = 0
        chain_onehot = chain_onehot.reindex(columns=[f'chain_{cid}' for cid in self.encoders['chain']], fill_value=0)
        
        # Token type one-hot
        token_onehot = pd.get_dummies(data['token_type'], prefix='token')
        for tt in self.encoders['token']:
            if f'token_{tt}' not in token_onehot.columns:
                token_onehot[f'token_{tt}'] = 0
        token_onehot = token_onehot.reindex(columns=[f'token_{tt}' for tt in self.encoders['token']], fill_value=0)
        
        # Gas fee
        gas_scaled = (np.log10(data['gas_fee'].apply(lambda x: max(x, 1e-8))) - 
                     self.scalers['gas']['mean']) / self.scalers['gas']['std']
        
        # TX hour
        hour_scaled = pd.to_datetime(data['timestamp']).dt.hour / 23.0
        
        # 결합
        features = pd.concat([
            amount_scaled.rename('amount_log'),
            time_gap_scaled.rename('time_gap'),
            function_onehot,
            risk_scaled.rename('risk_score'),
            chain_onehot,
            token_onehot,
            gas_scaled.rename('gas_fee'),
            hour_scaled.rename('tx_hour')
        ], axis=1)
        
        return features.values.astype(np.float32)


class SimpleAutoencoder(nn.Module):
    """간단한 Autoencoder 모델"""
    
    def __init__(self, input_dim=16, latent_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x):
        reconstructed, _ = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class WalletKeeperAnomalyDetector:
    """WalletKeeper 이상 탐지 시스템"""
    
    def __init__(self):
        self.encoder = SimpleTransactionFeatureEncoder()
        self.model = None
        self.threshold = 0.5
        self.is_trained = False
    
    def train(self, training_data: pd.DataFrame, epochs=50):
        """모델 학습"""
        logger.info("🚀 모델 학습 시작...")
        
        # Feature Engineering
        X = self.encoder.fit_transform(training_data)
        X_tensor = torch.FloatTensor(X)
        
        # 모델 생성
        self.model = SimpleAutoencoder(input_dim=X.shape[1], latent_dim=3)
        
        # 학습 설정
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # 학습 루프
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # 임계값 설정 (95 퍼센타일)
        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X_tensor)
            self.threshold = torch.quantile(errors, 0.95).item()
        
        self.is_trained = True
        logger.info(f"✅ 학습 완료! 임계값: {self.threshold:.6f}")
    
    def predict(self, transaction_data: Dict) -> Dict:
        """단일 트랜잭션 이상 탐지"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 데이터 전처리
        df = pd.DataFrame([transaction_data])
        X = self.encoder.transform(df)
        X_tensor = torch.FloatTensor(X)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            reconstructed, latent = self.model(X_tensor)
            mse_score = self.model.get_reconstruction_error(X_tensor).item()
            is_anomaly = mse_score > self.threshold
        
        return {
            'mse_score': mse_score,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'latent_representation': latent.numpy().tolist()[0],
            'input_features': X.tolist()[0]
        }


def create_sample_training_data(n_samples=1000) -> pd.DataFrame:
    """학습용 샘플 데이터 생성"""
    np.random.seed(42)
    
    # 정상 트랜잭션 데이터
    data = {
        'wallet_address': [f'0x{i:040x}' for i in np.random.randint(0, 100, n_samples)],
        'amount': np.random.lognormal(mean=2, sigma=1, size=n_samples),
        'timestamp': [
            datetime.now() - timedelta(hours=np.random.randint(0, 8760))  # 1년 내
            for _ in range(n_samples)
        ],
        'function_type': np.random.choice(['transfer', 'approve', 'swap', 'mint'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'counterparty_risk_score': np.random.beta(2, 8, n_samples),  # 대부분 낮은 위험도
        'chain_id': np.random.choice([1, 137, 56, 42161], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'token_type': np.random.choice(['ETH', 'ERC20', 'ERC721'], n_samples, p=[0.3, 0.6, 0.1]),
        'gas_fee': np.random.lognormal(mean=1, sigma=0.5, size=n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🚀 WalletKeeper Autoencoder 로컬 테스트 시스템")
    print("=" * 60)
    
    # 간단한 테스트
    detector = WalletKeeperAnomalyDetector()
    training_data = create_sample_training_data(500)
    detector.train(training_data, epochs=30)
    
    # 테스트 트랜잭션
    test_tx = {
        'wallet_address': '0x1234567890abcdef1234567890abcdef12345678',
        'amount': 100.0,
        'timestamp': datetime.now().isoformat(),
        'function_type': 'transfer',
        'counterparty_risk_score': 0.05,
        'chain_id': 1,
        'token_type': 'ETH',
        'gas_fee': 0.002
    }
    
    result = detector.predict(test_tx)
    print(f"\n🧪 테스트 결과:")
    print(f"이상 여부: {result['is_anomaly']}")
    print(f"MSE 점수: {result['mse_score']:.6f}")
    print(f"임계값: {result['threshold']:.6f}")