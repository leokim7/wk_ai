#!/usr/bin/env python3
"""
TX Hash에서 트랜잭션 데이터를 자동 추출하는 시스템
Infura, Alchemy 등 Web3 Provider 지원
"""

import os
import requests
import json
from datetime import datetime
from typing import Dict, Optional, List
import logging
from web3 import Web3
from eth_utils import to_checksum_address
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionDataFetcher:
    """
    TX Hash에서 트랜잭션 데이터를 추출하는 클래스
    """
    
    def __init__(self, infura_api_key: str = None, alchemy_api_key: str = None):
        """
        초기화
        
        Args:
            infura_api_key: Infura API 키
            alchemy_api_key: Alchemy API 키
        """
        self.infura_api_key = infura_api_key or os.getenv('INFURA_API_KEY')
        self.alchemy_api_key = alchemy_api_key or os.getenv('ALCHEMY_API_KEY')
        
        # Web3 provider 설정 (우선순위: Alchemy > Infura > Public)
        self.w3_providers = {}
        self._setup_providers()
        
        # 체인별 설정
        self.chain_configs = {
            1: {
                'name': 'Ethereum Mainnet',
                'currency': 'ETH',
                'decimals': 18
            },
            137: {
                'name': 'Polygon',
                'currency': 'MATIC',
                'decimals': 18
            },
            56: {
                'name': 'BSC',
                'currency': 'BNB',
                'decimals': 18
            },
            42161: {
                'name': 'Arbitrum One',
                'currency': 'ETH',
                'decimals': 18
            }
        }
        
        # 함수 시그니처 매핑
        self.function_signatures = {
            '0xa9059cbb': 'transfer',
            '0x095ea7b3': 'approve',
            '0x23b872dd': 'transferFrom',
            '0x40c10f19': 'mint',
            '0x42842e0e': 'safeTransferFrom',
            '0xa22cb465': 'setApprovalForAll',
            '0x': 'transfer',  # ETH 전송
            None: 'transfer'   # 기본값
        }
    
    def _setup_providers(self):
        """Web3 provider 설정"""
        try:
            # Ethereum Mainnet
            if self.alchemy_api_key:
                self.w3_providers[1] = Web3(Web3.HTTPProvider(
                    f'https://eth-mainnet.alchemyapi.io/v2/{self.alchemy_api_key}'
                ))
                logger.info("✅ Alchemy Ethereum provider 설정 완료")
            elif self.infura_api_key:
                self.w3_providers[1] = Web3(Web3.HTTPProvider(
                    f'https://mainnet.infura.io/v3/{self.infura_api_key}'
                ))
                logger.info("✅ Infura Ethereum provider 설정 완료")
            else:
                # Public endpoint (제한적)
                self.w3_providers[1] = Web3(Web3.HTTPProvider('https://cloudflare-eth.com'))
                logger.warning("⚠️ Public Ethereum provider 사용 (API 키 권장)")
            
            # Polygon
            if self.alchemy_api_key:
                self.w3_providers[137] = Web3(Web3.HTTPProvider(
                    f'https://polygon-mainnet.alchemyapi.io/v2/{self.alchemy_api_key}'
                ))
            elif self.infura_api_key:
                self.w3_providers[137] = Web3(Web3.HTTPProvider(
                    f'https://polygon-mainnet.infura.io/v3/{self.infura_api_key}'
                ))
            else:
                self.w3_providers[137] = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
            
            # BSC
            self.w3_providers[56] = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org'))
            
            # Arbitrum One
            if self.alchemy_api_key:
                self.w3_providers[42161] = Web3(Web3.HTTPProvider(
                    f'https://arb-mainnet.alchemyapi.io/v2/{self.alchemy_api_key}'
                ))
            else:
                self.w3_providers[42161] = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
            
        except Exception as e:
            logger.error(f"Provider 설정 실패: {e}")
    
    def fetch_transaction_data(self, tx_hash: str, chain_id: int = 1, 
                             counterparty_risk_score: float = None) -> Dict:
        """
        TX Hash에서 트랜잭션 데이터 추출
        
        Args:
            tx_hash: 트랜잭션 해시
            chain_id: 체인 ID (기본값: 1 = Ethereum)
            counterparty_risk_score: 수동 입력 위험도 점수
            
        Returns:
            WalletKeeper에 필요한 8개 필드 데이터
        """
        try:
            logger.info(f"🔍 TX Hash 분석 시작: {tx_hash} (Chain: {chain_id})")
            
            # Web3 provider 가져오기
            if chain_id not in self.w3_providers:
                raise ValueError(f"지원하지 않는 체인 ID: {chain_id}")
            
            w3 = self.w3_providers[chain_id]
            
            # 트랜잭션 정보 가져오기
            tx = w3.eth.get_transaction(tx_hash)
            tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
            
            # 블록 정보 가져오기 (timestamp)
            block = w3.eth.get_block(tx.blockNumber)
            
            # 1. wallet_address (from 주소)
            wallet_address = to_checksum_address(tx['from'])
            
            # 2. amount (Wei에서 Ether로 변환)
            amount = float(w3.from_wei(tx.value, 'ether'))
            
            # 3. timestamp (블록 타임스탬프에서 변환)
            timestamp = datetime.fromtimestamp(block.timestamp).isoformat() + 'Z'
            
            # 4. function_type (input data에서 함수 시그니처 추출)
            function_type = self._extract_function_type(tx.input)
            
            # 5. counterparty_risk_score (수동 입력 또는 기본값)
            if counterparty_risk_score is None:
                counterparty_risk_score = self._estimate_risk_score(tx, tx_receipt)
            
            # 6. chain_id (입력받은 값)
            # 이미 있음
            
            # 7. token_type 결정
            token_type = self._determine_token_type(tx, tx_receipt)
            
            # 8. gas_fee (가스 사용량 * 가스 가격)
            gas_fee = float(w3.from_wei(tx_receipt.gasUsed * tx.gasPrice, 'ether'))
            
            # 결과 구성
            result = {
                'wallet_address': wallet_address,
                'amount': amount,
                'timestamp': timestamp,
                'function_type': function_type,
                'counterparty_risk_score': counterparty_risk_score,
                'chain_id': chain_id,
                'token_type': token_type,
                'gas_fee': gas_fee,
                # 추가 메타데이터
                'meta': {
                    'tx_hash': tx_hash,
                    'to_address': to_checksum_address(tx.to) if tx.to else None,
                    'block_number': tx.blockNumber,
                    'gas_used': tx_receipt.gasUsed,
                    'gas_price': tx.gasPrice,
                    'chain_name': self.chain_configs.get(chain_id, {}).get('name', f'Chain {chain_id}')
                }
            }
            
            logger.info(f"✅ 트랜잭션 데이터 추출 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ 트랜잭션 데이터 추출 실패: {e}")
            raise
    
    def _extract_function_type(self, input_data: str) -> str:
        """입력 데이터에서 함수 타입 추출"""
        if not input_data or input_data == '0x':
            return 'transfer'  # ETH 전송
        
        # 함수 시그니처 (처음 4바이트)
        function_selector = input_data[:10] if len(input_data) >= 10 else input_data
        
        return self.function_signatures.get(function_selector, 'unknown')
    
    def _determine_token_type(self, tx, tx_receipt) -> str:
        """토큰 타입 결정"""
        # ETH 전송인 경우
        if tx.value > 0 and (not tx.input or tx.input == '0x'):
            return 'ETH'
        
        # 컨트랙트 상호작용 분석
        if tx.to:
            # 로그 분석으로 토큰 타입 추정
            for log in tx_receipt.logs:
                topics = [topic.hex() for topic in log.topics]
                
                # ERC20 Transfer 이벤트
                if len(topics) >= 3 and topics[0] == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                    # Transfer(address,address,uint256)
                    if len(topics) == 3:  # ERC20
                        return 'ERC20'
                    elif len(topics) == 4:  # ERC721
                        return 'ERC721'
                
                # ERC721 Transfer 이벤트 (추가 확인)
                if (len(topics) == 4 and 
                    topics[0] == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'):
                    return 'ERC721'
            
            # 기본값은 ERC20 (컨트랙트 상호작용)
            return 'ERC20'
        
        return 'ETH'  # 기본값
    
    def _estimate_risk_score(self, tx, tx_receipt) -> float:
        """위험도 점수 추정 (간단한 휴리스틱)"""
        risk_score = 0.0
        
        # 기본 위험도 (매우 낮음)
        risk_score += 0.01
        
        # 가스 가격이 높으면 위험도 증가
        if tx.gasPrice > Web3.to_wei(50, 'gwei'):
            risk_score += 0.02
        
        # 트랜잭션이 실패했으면 위험도 증가
        if tx_receipt.status == 0:
            risk_score += 0.1
        
        # 새로운 컨트랙트와의 상호작용
        if tx.to and tx_receipt.contractAddress:
            risk_score += 0.05
        
        # 복잡한 트랜잭션 (많은 로그)
        if len(tx_receipt.logs) > 5:
            risk_score += 0.03
        
        # 최대값 제한
        return min(risk_score, 1.0)


class WalletKeeperWithTxHash:
    """
    TX Hash 기반 WalletKeeper 이상 탐지 시스템
    """
    
    def __init__(self, infura_api_key: str = None, alchemy_api_key: str = None):
        self.fetcher = TransactionDataFetcher(infura_api_key, alchemy_api_key)
        self.detector = None  # WalletKeeperAnomalyDetector 인스턴스
    
    def setup_detector(self, training_data_size: int = 500):
        """이상 탐지 모델 설정"""
        from local_test_system import WalletKeeperAnomalyDetector, create_sample_training_data
        
        logger.info("🧠 이상 탐지 모델 초기화 중...")
        self.detector = WalletKeeperAnomalyDetector()
        
        # 학습 데이터 생성 및 학습
        training_data = create_sample_training_data(training_data_size)
        self.detector.train(training_data, epochs=30)
        
        logger.info("✅ 이상 탐지 모델 준비 완료")
    
    def analyze_transaction_by_hash(self, tx_hash: str, chain_id: int = 1, 
                                  counterparty_risk_score: float = None) -> Dict:
        """
        TX Hash로 트랜잭션 분석
        
        Args:
            tx_hash: 트랜잭션 해시
            chain_id: 체인 ID
            counterparty_risk_score: 수동 위험도 점수 (선택사항)
            
        Returns:
            이상 탐지 결과 + 원본 트랜잭션 데이터
        """
        if not self.detector:
            raise ValueError("먼저 setup_detector()를 호출하세요.")
        
        # 1. TX Hash에서 데이터 추출
        tx_data = self.fetcher.fetch_transaction_data(tx_hash, chain_id, counterparty_risk_score)
        
        # 2. 이상 탐지 수행
        anomaly_result = self.detector.predict(tx_data)
        
        # 3. 결과 통합
        result = {
            'transaction_data': tx_data,
            'anomaly_detection': anomaly_result,
            'summary': {
                'tx_hash': tx_hash,
                'is_anomaly': anomaly_result['is_anomaly'],
                'risk_level': self._calculate_risk_level(anomaly_result),
                'mse_score': anomaly_result['mse_score'],
                'threshold': anomaly_result['threshold']
            }
        }
        
        return result
    
    def _calculate_risk_level(self, anomaly_result: Dict) -> str:
        """위험도 레벨 계산"""
        mse_score = anomaly_result['mse_score']
        threshold = anomaly_result['threshold']
        
        if mse_score < threshold * 0.5:
            return 'LOW'
        elif mse_score < threshold:
            return 'MEDIUM'
        elif mse_score < threshold * 2:
            return 'HIGH'
        else:
            return 'CRITICAL'


if __name__ == "__main__":
    print("🔧 필요한 패키지 설치:")
    print("pip install web3 eth-utils requests")
    print("\n환경 변수 설정 (선택사항):")
    print("export INFURA_API_KEY='your_infura_key'")
    print("export ALCHEMY_API_KEY='your_alchemy_key'")
    print("\n" + "=" * 60)
    
    # 간단한 테스트
    analyzer = WalletKeeperWithTxHash()
    analyzer.setup_detector(training_data_size=300)
    
    print("🧪 테스트용 더미 분석...")
    test_hash = "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060"
    try:
        result = analyzer.analyze_transaction_by_hash(test_hash, chain_id=1)
        print(f"결과: {'이상' if result['summary']['is_anomaly'] else '정상'}")
    except Exception as e:
        print(f"테스트 실패 (API 키 없음): {e}")