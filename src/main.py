import wandb 
from tqdm import tqdm
import torch
import omegaconf
from utils import load_config
from model import EncoderForClassification
from data import get_dataloader

def train_iter(model, inputs, optimizer, device, epoch):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({'train_loss' : loss.item()})
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

    for model_name, model_config, data_config in experiments:
        model = EncoderForClassification(model_config).to(device)
        train_loader = get_dataloader(data_config, split='train')
        valid_loader = get_dataloader(data_config, split='valid')
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.train_config.learning_rate)

        wandb.login(key=configs.wandb_config.api_key)
        wandb.init(
            project=configs.wandb_config.project,
            name=f"[imdb] {model_name}",
        )
        
        for epoch in range(configs.train_config.epochs) :
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.train_config.epochs}")
            for step, inputs in enumerate(pbar):
                loss = train_iter(model, inputs, optimizer, device, epoch)
                pbar.set_postfix({'loss': loss.item()})

                if (step + 1) % configs.train_config.log_steps == 0:
                    wandb.log({'step': step + 1, 'epoch': epoch + 1})

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
                'val_accuracy': avg_val_accuracy
            })
            print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        wandb.finish()  

if __name__ == "__main__" :
    configs = load_config()
    main(configs)
