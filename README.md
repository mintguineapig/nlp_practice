# BERT vs ModernBERT Comparison

IMDB 영화 리뷰 감정 분석을 위한 BERT와 ModernBERT 성능 비교 실험

## 🎯 프로젝트 개요

- **데이터셋**: IMDB 영화 리뷰 (긍정/부정 감정 분류)
- **모델**: BERT-base-uncased vs answerdotai/ModernBERT-base
- **실험 추적**: Weights & Biases (WandB)
- **환경**: Python 3.9, PyTorch, Transformers

## 📁 프로젝트 구조

```
exp_1/
├── src/
│   ├── main.py          # 메인 실험 실행 스크립트
│   ├── model.py         # 분류 모델 정의
│   ├── data.py          # 데이터 로딩 및 전처리 (사전 토크나이징)
│   └── utils.py         # 설정 로딩 유틸리티
├── configs/
│   └── configs.yaml     # 모델, 데이터, 훈련 설정
├── wandb/              # WandB 실험 로그 (git에서 제외)
└── README.md
```

## 🚀 주요 특징

### 1. **사전 토크나이징 (Pre-tokenization)**
- 훈련 중 토크나이징 오버헤드 완전 제거
- 빠른 데이터 로딩으로 훈련 속도 향상

### 2. **간결한 코드 설계**
- 핵심 기능만 포함된 최소한의 코드
- 명확한 타입 힌트와 docstring
- 한 줄 `__getitem__` 구현

### 3. **설정 기반 실험 관리**
- OmegaConf를 활용한 YAML 설정 파일
- 코드 수정 없이 하이퍼파라미터 변경 가능
- 모델별 독립적인 설정 관리

### 4. **자동화된 실험 비교**
- 두 모델 순차적 훈련 및 비교
- WandB를 통한 실시간 실험 추적
- 검증 성능 자동 로깅

## ⚙️ 설치 및 실행

### 환경 설정
```bash
# Python 3.9 가상환경 생성
python3.9 -m venv venv39
source venv39/bin/activate

# 필요한 패키지 설치
pip install torch tqdm wandb datasets numpy omegaconf "transformers>=4.46.0"
```

### 실험 실행
```bash
cd exp_1
python src/main.py
```

## 📊 실험 설정

### 모델 설정
- **BERT**: bert-base-uncased (768 hidden size)
- **ModernBERT**: answerdotai/ModernBERT-base (768 hidden size)

### 데이터 설정
- **최대 길이**: 128 토큰
- **배치 크기**: 32
- **분할**: Train(45,000) / Valid(2,500) / Test(2,500)

### 훈련 설정
- **에포크**: 5
- **학습률**: 5e-5
- **옵티마이저**: Adam
- **손실 함수**: CrossEntropyLoss

## 🎯 핵심 구현 사항

### 1. 사전 토크나이징 데이터셋
```python
def __getitem__(self, idx: int) -> dict:
    return {key: self.dataset[key][idx] for key in self.dataset}
```

### 2. 간결한 모델 정의
```python
def forward(self, input_ids, attention_mask, label, token_type_ids=None):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state 
    pooled = hidden[:, 0, :]  # CLS 토큰 사용
    logits = self.classifier(pooled)
    loss = self.loss_fn(logits, label)
    return {'logits': logits, 'loss': loss}
```

### 3. 효율적인 배치 처리
```python
@staticmethod
def collate_fn(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
```

## 📈 실험 결과

WandB 대시보드에서 다음 메트릭을 확인할 수 있습니다:
- 훈련/검증 손실
- 검증 정확도
- 에포크별 성능 비교
- 모델별 수렴 속도

## 🛠️ 기술 스택

- **Python**: 3.9+
- **PyTorch**: 2.8.0
- **Transformers**: 4.55.2 (ModernBERT 지원)
- **Datasets**: 4.0.0 (IMDB 데이터 로딩)
- **WandB**: 0.21.1 (실험 추적)
- **OmegaConf**: 2.3.0 (설정 관리)


