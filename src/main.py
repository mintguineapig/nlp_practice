import wandb 
from tqdm import tqdm
import torch
import omegaconf
import time
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler  # 혼합 정밀도 훈련을 위한 임포트
from utils import load_config
from model import EncoderForClassification
from data import get_dataloader
import gc

def train_iter(model, inputs, optimizer, scheduler, scaler, device, epoch, step, accumulation_steps, is_last_step=False):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    
    # 혼합 정밀도 사용
    with autocast():
        outputs = model(**inputs)
        loss = outputs['loss']
    
    # 손실을 accumulation_steps로 나누어 스케일링 (더 작은 업데이트 단계)
    scaled_loss = loss / accumulation_steps
    
    # 그래디언트 스케일링과 함께 역전파
    scaler.scale(scaled_loss).backward()
    
    # accumulation_steps 마다 가중치 업데이트 또는 마지막 배치일 때
    if (step + 1) % accumulation_steps == 0 or is_last_step:
        # 그래디언트 unscale 및 스텝
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    
    wandb.log({'train_loss' : loss.item()})  # 원래 손실값 기록
    return loss

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs : omegaconf.DictConfig) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiments = [
        ("ModernBERT", configs.model_config_modernbert, configs.data_config_modernbert),
        ("BERT", configs.model_config_bert, configs.data_config_bert)
    ]
    
    # 전체 시작 시간 기록
    total_start_time = time.time()

    for model_name, model_config, data_config in experiments:
        # 모델별 시작 시간 기록
        model_start_time = time.time()
        
        model = EncoderForClassification(model_config).to(device)
        train_loader = get_dataloader(data_config, split='train')
        valid_loader = get_dataloader(data_config, split='valid')
        
        # 혼합 정밀도 훈련을 위한 그래디언트 스케일러 생성
        scaler = GradScaler()
        
        # Setup optimizer based on config
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.train_config.learning_rate)
        
        # Setup scheduler based on config
        scheduler = None
        if configs.train_config.scheduler == "constant":
            # Lambda function returns 1.0 for constant learning rate
            scheduler = LambdaLR(optimizer, lambda _: 1.0)
        
        # Gradient Accumulation 설정
        accumulation_steps = 4  # 기본값으로 4 설정
        
        # data_config에 accumulation_steps가 있으면 사용
        if hasattr(configs.train_config, 'accumulation_steps'):
            accumulation_steps = configs.train_config.accumulation_steps
        
        # 실질적인 배치 크기 계산
        effective_batch_size = data_config.batch_size * accumulation_steps
            
        # Log hyperparameters
        wandb.login(key=configs.wandb_config.api_key)
        wandb.init(
            project=configs.wandb_config.project,
            name=f"[imdb] {model_name} - bs{data_config.batch_size} (eff_bs{effective_batch_size})",
            config={
                "learning_rate": configs.train_config.learning_rate,
                "scheduler": configs.train_config.scheduler,
                "optimizer": configs.train_config.optimizer,
                "max_length": data_config.max_length,
                "batch_size": data_config.batch_size,
                "effective_batch_size": effective_batch_size,
                "accumulation_steps": accumulation_steps,
                "epochs": configs.train_config.epochs,
                "model": model_name
            }
        )
        
        # 실질적인 배치 크기 출력
        print(f"{model_name}의 실질적인 배치 크기: {effective_batch_size} (배치 크기: {data_config.batch_size} x 누적 단계: {accumulation_steps})")
            
        for epoch in range(configs.train_config.epochs) :
            epoch_start_time = time.time()
            model.train()
            
            # 학습 시작 시 그래디언트 초기화
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.train_config.epochs}")
            for step, inputs in enumerate(pbar):
                # 마지막 배치 여부 확인
                is_last_step = (step == len(train_loader) - 1)
                
                loss = train_iter(model, inputs, optimizer, scheduler, scaler, device, epoch, step, accumulation_steps, is_last_step)
                pbar.set_postfix({'loss': loss.item()})

                if (step + 1) % configs.train_config.log_steps == 0:
                    # 현재 학습률 기록
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb.log({'step': step + 1, 'epoch': epoch + 1, 'learning_rate': current_lr})

            model.eval()
            val_losses = []
            val_accuracies = []
            with torch.no_grad():
                for inputs in valid_loader:
                    val_loss, val_accuracy = valid_iter(model, inputs, device)
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy)
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            wandb.log({
                'epoch': epoch + 1,
                'val_loss': avg_val_loss, 
                'val_accuracy': avg_val_accuracy,
                'epoch_time': time.time() - epoch_start_time
            })
            print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, 시간: {time.time() - epoch_start_time:.2f}초")
        
        # 배치 처리 후 불필요한 캐시 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        # 모델별 종료 시간 기록 및 소요 시간 계산
        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        wandb.log({"total_training_time": model_duration})
        print(f"{model_name} 학습 완료 - 총 소요 시간: {model_duration:.2f}초 ({model_duration/60:.2f}분)")
        
        wandb.finish()  

if __name__ == "__main__" :
    # 설정 파일 로드 (utils.py에서 명령줄 인수 처리)
    configs = load_config()
    
    # 배치 크기 출력
    print(f"ModernBERT 배치 크기: {configs.data_config_modernbert.batch_size}")
    print(f"BERT 배치 크기: {configs.data_config_bert.batch_size}")
    
    # 메인 실행
    start_time = time.time()
    main(configs)
    print(f"전체 실험 완료 - 총 소요 시간: {time.time() - start_time:.2f}초 ({(time.time() - start_time)/60:.2f}분)")
