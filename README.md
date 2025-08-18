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

<img width="481" height="389" alt="image" src="https://github.com/user-attachments/assets/b641d387-eba9-4785-8efc-37f789da8d60" />

modernBERT의 성능이 BERT의 성능보다 더 좋다.

modernBERT 는 BERT 와 유사한 파라미터 수를 갖지만, 미세하게 토크나이저와 같은 세부적인 아키텍처가 최적화되어있기 때문이다. 

(출처 : https://jina.ai/ko/news/what-should-we-learn-from-modernbert/ )

이로서 scaling law 로 파라미터 수를 늘리는 것 뿐만 아니라 아키텍처 최적화도 성능 향상을 위해 필요하다는 점을 알 수 있다.


## 🧑🏼‍💻 코드 작성 시 주의할 점
1. Configs 를 활용하여 효율적으로 파라미터 조절

2. input, output 자료형을 함수에 명시하여 오류를 예방

3. 속도를 개선하기 위해 사전 tokenizing 을 수행

4. Reproductivity 를 위한 seed 고정 (seed = 42)

5. wandb API 키는 본인의 key 로 대체해야 합니다. 

## 🛠️ 기술 스택

- **Python**: 3.9+
- **PyTorch**: 2.8.0
- **Transformers**: 4.55.2 (ModernBERT 지원)
- **Datasets**: 4.0.0 (IMDB 데이터 로딩)
- **WandB**: 0.21.1 (실험 추적)
- **OmegaConf**: 2.3.0 (설정 관리)


