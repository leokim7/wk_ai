#!/usr/bin/env python3
"""
WalletKeeper 웹 애플리케이션 백엔드
FastAPI + HTML 템플릿
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import logging

from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
analyzer = None

class TxAnalysisRequest(BaseModel):
    """TX 분석 요청 스키마"""
    tx_hash: str = Field(..., description="트랜잭션 해시")
    chain_id: int = Field(1, description="체인 ID")
    risk_score: Optional[float] = Field(None, ge=0, le=1, description="위험도 점수")

class TxAnalysisResponse(BaseModel):
    """TX 분석 응답 스키마"""
    success: bool
    tx_hash: str
    chain_name: str
    transaction_data: Dict
    anomaly_result: Dict
    summary: Dict
    processing_time: float
    timestamp: str
    error_message: Optional[str] = None

# FastAPI 앱 초기화
app = FastAPI(
    title="WalletKeeper Web Interface",
    description="TX Hash 기반 이상 탐지 웹 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모델 초기화"""
    global analyzer
    
    logger.info("🚀 WalletKeeper 웹 애플리케이션 시작 중...")
    
    try:
        # TX Hash 분석기 초기화
        from tx_hash_fetcher import WalletKeeperWithTxHash
        
        # 환경 변수에서 API 키 로드
        infura_key = os.getenv('INFURA_API_KEY')
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        
        if not infura_key and not alchemy_key:
            logger.warning("⚠️ API 키가 설정되지 않음. Public endpoint 사용")
        
        analyzer = WalletKeeperWithTxHash(infura_key, alchemy_key)
        analyzer.setup_detector(training_data_size=300)  # 웹용으로 빠른 초기화
        
        logger.info("✅ 모델 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        # 더미 분석기로 대체
        analyzer = DummyAnalyzer()

class DummyAnalyzer:
    """테스트용 더미 분석기"""
    
    def analyze_transaction_by_hash(self, tx_hash: str, chain_id: int = 1, 
                                  counterparty_risk_score: float = None):
        import random
        import time
        
        time.sleep(1)  # 실제 분석 시뮬레이션
        
        # 더미 데이터 생성
        is_anomaly = random.choice([True, False])
        mse_score = random.uniform(0.01, 0.3)
        threshold = 0.15
        
        chain_names = {1: "Ethereum", 137: "Polygon", 56: "BSC", 42161: "Arbitrum"}
        
        return {
            'transaction_data': {
                'wallet_address': f'0x{random.randint(0, 2**160):040x}',
                'amount': random.uniform(0.1, 10000),
                'timestamp': datetime.now().isoformat() + 'Z',
                'function_type': random.choice(['transfer', 'approve', 'swap', 'mint']),
                'counterparty_risk_score': counterparty_risk_score or random.uniform(0.01, 0.5),
                'chain_id': chain_id,
                'token_type': random.choice(['ETH', 'ERC20', 'ERC721']),
                'gas_fee': random.uniform(0.001, 0.1),
                'meta': {
                    'tx_hash': tx_hash,
                    'chain_name': chain_names.get(chain_id, f'Chain {chain_id}'),
                    'block_number': random.randint(18000000, 19000000)
                }
            },
            'anomaly_detection': {
                'mse_score': mse_score,
                'is_anomaly': is_anomaly,
                'threshold': threshold,
                'latent_representation': [random.uniform(-1, 1) for _ in range(3)]
            },
            'summary': {
                'tx_hash': tx_hash,
                'is_anomaly': is_anomaly,
                'risk_level': 'HIGH' if is_anomaly else 'LOW',
                'mse_score': mse_score,
                'threshold': threshold
            }
        }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """홈페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze", response_model=TxAnalysisResponse)
async def analyze_transaction(request: TxAnalysisRequest):
    """TX Hash 분석 API"""
    start_time = datetime.now()
    
    try:
        # 입력 검증
        tx_hash = request.tx_hash.strip()
        if not tx_hash.startswith('0x') or len(tx_hash) != 66:
            raise HTTPException(
                status_code=400,
                detail="올바른 TX Hash 형식이 아닙니다 (0x로 시작하는 66자)"
            )
        
        if request.chain_id not in [1, 137, 56, 42161]:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 체인 ID입니다"
            )
        
        # 분석 수행
        logger.info(f"🔍 TX 분석 시작: {tx_hash}")
        
        result = analyzer.analyze_transaction_by_hash(
            tx_hash=tx_hash,
            chain_id=request.chain_id,
            counterparty_risk_score=request.risk_score
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 체인 이름 가져오기
        chain_names = {
            1: "Ethereum Mainnet",
            137: "Polygon",
            56: "BSC",
            42161: "Arbitrum One"
        }
        
        response = TxAnalysisResponse(
            success=True,
            tx_hash=tx_hash,
            chain_name=chain_names.get(request.chain_id, f"Chain {request.chain_id}"),
            transaction_data=result['transaction_data'],
            anomaly_result=result['anomaly_detection'],
            summary=result['summary'],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ TX 분석 완료: {tx_hash} ({'이상' if result['summary']['is_anomaly'] else '정상'})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TX 분석 실패: {tx_hash}, 오류: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TxAnalysisResponse(
            success=False,
            tx_hash=request.tx_hash,
            chain_name="Unknown",
            transaction_data={},
            anomaly_result={},
            summary={},
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            error_message=str(e)
        )

@app.post("/analyze-form")
async def analyze_form(
    request: Request,
    tx_hash: str = Form(...),
    chain_id: int = Form(1),
    risk_score: Optional[str] = Form(None)
):
    try:
        # risk_score 처리 - 빈 문자열이면 None으로 변환
        processed_risk_score = None
        if risk_score and risk_score.strip():
            try:
                processed_risk_score = float(risk_score)
                # 0-1 범위 체크
                if processed_risk_score < 0 or processed_risk_score > 1:
                    processed_risk_score = None
            except (ValueError, TypeError):
                processed_risk_score = None
        
        # API 호출
        analysis_request = TxAnalysisRequest(
            tx_hash=tx_hash,
            chain_id=chain_id,
            risk_score=processed_risk_score
        )
        
        result = await analyze_transaction(analysis_request)
        
        # 결과 페이지 렌더링
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result.dict()
        })
        
    except HTTPException as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": e.detail,
            "tx_hash": tx_hash
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"예상치 못한 오류: {str(e)}",
            "tx_hash": tx_hash
        })
    """폼 기반 TX 분석 (HTML 폼에서 호출)"""
    try:
        # API 호출
        analysis_request = TxAnalysisRequest(
            tx_hash=tx_hash,
            chain_id=chain_id,
            risk_score=risk_score
        )
        
        result = await analyze_transaction(analysis_request)
        
        # 결과 페이지 렌더링
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result.dict()
        })
        
    except HTTPException as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": e.detail,
            "tx_hash": tx_hash
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"예상치 못한 오류: {str(e)}",
            "tx_hash": tx_hash
        })

@app.get("/batch", response_class=HTMLResponse)
async def batch_analysis_page(request: Request):
    """배치 분석 페이지"""
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/api/batch-analyze")
async def batch_analyze(tx_hashes: List[str], chain_id: int = 1):
    """배치 TX 분석 API"""
    if len(tx_hashes) > 10:  # 웹에서는 최대 10개로 제한
        raise HTTPException(
            status_code=400,
            detail="한 번에 최대 10개 트랜잭션만 분석 가능합니다"
        )
    
    results = []
    
    for tx_hash in tx_hashes:
        try:
            analysis_request = TxAnalysisRequest(
                tx_hash=tx_hash.strip(),
                chain_id=chain_id
            )
            result = await analyze_transaction(analysis_request)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "success": False,
                "tx_hash": tx_hash,
                "error_message": str(e)
            })
    
    return {"results": results, "total": len(tx_hashes)}

@app.get("/examples", response_class=HTMLResponse) 
async def examples_page(request: Request):
    """예시 페이지"""
    
    # 실제 TX Hash 예시들
    examples = [
        {
            "name": "정상적인 ETH 전송",
            "tx_hash": "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060",
            "chain_id": 1,
            "description": "일반적인 이더리움 전송 트랜잭션",
            "expected": "정상"
        },
        {
            "name": "ERC20 토큰 전송", 
            "tx_hash": "0xa9059cbb2ab09eb219583f4a59a5d0623ade346d962bcd4e46b11da047c9049b",
            "chain_id": 1,
            "description": "USDC 등 ERC20 토큰 전송",
            "expected": "정상"
        },
        {
            "name": "DeFi 스왑 트랜잭션",
            "tx_hash": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
            "chain_id": 1,
            "description": "Uniswap 등에서의 토큰 스왑",
            "expected": "정상"
        }
    ]
    
    return templates.TemplateResponse("examples.html", {
        "request": request,
        "examples": examples
    })

if __name__ == "__main__":
    import uvicorn
    
    # 디렉토리 생성
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    print("🚀 WalletKeeper 웹 애플리케이션 시작")
    print("📱 브라우저에서 http://localhost:8000 접속")
    print("📚 API 문서: http://localhost:8000/docs")
    
    uvicorn.run(
        "web_app_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )