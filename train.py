from dataloader import GolfDB, Normalize, ToTensor, ColorJitter, RandomErasing
from model import EventDetector
from util import *
from eval import eval
from training_utils import (
    FocalLoss, EMA, SWA, mixup_data, mixup_criterion, init_weights,
    Lookahead, WarmupScheduler
)
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import time
import numpy as np


if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 15000
    it_save = 100  # save model every 100 iterations
    it_eval = 200  # evaluate on validation set every N iterations
    n_cpu = 8  # Increased from 6 for better data loading parallelism
    seq_length = 64
    bs = 44  # Increased from 22 (doubled) - FP16 allows larger batches
    k = 10  # frozen layers
    
    # Early stopping configuration
    early_stop_patience = 20  # Stop if no improvement for N validation checks
    early_stop_min_delta = 0.001  # Minimum improvement to consider as better
    best_pce = 0.0
    no_improve_count = 0
    best_iteration = 0

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=True)  # Enabled dropout for regularization
    freeze_layers(k, model)
    model.train()
    model.cuda()
    
    # Initialize weights (only for non-pretrained parts)
    # Note: MobileNetV2 is pretrained, so we only init LSTM and linear layers
    for name, module in model.named_modules():
        if 'rnn' in name or 'lin' in name:
            module.apply(init_weights)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # Using reduce-overhead mode for better performance
    try:
        import warnings
        # Suppress Dynamo warnings related to Variable compatibility
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
            model = torch.compile(model, mode="reduce-overhead")
        print("Model compilation: ENABLED (torch.compile, reduce-overhead mode)")
    except Exception as e:
        print(f"Model compilation: DISABLED ({e})")
        # Continue without compilation if not supported

    # Data augmentation: ColorJitter for training (light augmentation to preserve swing mechanics)
    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='/data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([
                         ToTensor(),
                         ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
                     train=True)

    # Optimized data loading: pin_memory for faster CPU→GPU transfer, prefetch_factor for pre-loading
    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             pin_memory=True,  # Faster CPU→GPU memory transfer
                             prefetch_factor=4,  # Pre-load next 4 batches (increased)
                             persistent_workers=True,  # Keep workers alive between epochs
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    # Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weights)
    # AdamW optimizer (better weight decay handling, fused operations)
    base_optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.002,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Lookahead optimizer wrapper (better convergence stability)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    # Learning rate scheduling: Cosine Annealing with Warm Restarts
    base_scheduler = CosineAnnealingWarmRestarts(
        base_optimizer,  # Use base optimizer for scheduler
        T_0=2000,  # First restart period
        T_mult=2,  # Multiplier for restart period
        eta_min=1e-6
    )
    
    # Add warmup to scheduler
    warmup_steps = 500  # Warm up over 500 iterations
    scheduler = WarmupScheduler(base_scheduler, warmup_steps=warmup_steps)
    
    # Gradient accumulation configuration
    accumulation_steps = 2  # Accumulate gradients over 2 batches
    
    # Exponential Moving Average
    ema = EMA(model, decay=0.9999)
    
    # Stochastic Weight Averaging (complement to EMA, activates later in training)
    swa_start = int(iterations * 0.75)  # Start SWA at 75% of training
    swa = SWA(model, swa_start=swa_start, swa_freq=50, swa_lr=1e-5)

    # Mixed precision training: Use FP16 for faster training on A100 GPUs
    # GradScaler prevents gradient underflow in FP16
    scaler = GradScaler()

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    # Cost monitoring: Modal A100 40GB = $0.000583/sec = $2.10/hour
    # Modal will timeout after 9.5 hours (set in modal_runner.py) to stay under $20 budget
    ESTIMATED_COST_PER_HOUR = 2.10
    MAX_BUDGET = 20.0
    start_time = time.time()
    
    print('='*60)
    print('Starting training with validation monitoring')
    print('='*60)
    print('OPTIMIZATIONS ENABLED:')
    print('  - Mixed Precision Training: ENABLED (FP16)')
    print('  - Optimizer: Lookahead(AdamW) (k=5, alpha=0.5, better convergence)')
    print('  - Dropout: ENABLED (regularization)')
    print('  - Weight Decay: ENABLED (1e-4, L2 regularization)')
    print('  - Learning Rate Scheduling: ENABLED (Warmup + Cosine Annealing with Warm Restarts)')
    print('    * Warmup: 500 iterations (linear warmup)')
    print('    * Cosine Annealing: T_0=2000, T_mult=2, eta_min=1e-6')
    print('  - Gradient Clipping: ENABLED (max_norm=1.0)')
    print('  - Gradient Accumulation: ENABLED (steps=2)')
    print('  - Exponential Moving Average (EMA): ENABLED (decay=0.9999)')
    print('  - Stochastic Weight Averaging (SWA): ENABLED (starts at {} iterations)'.format(swa_start))
    print('  - Focal Loss: ENABLED (alpha=1.0, gamma=2.0, for class imbalance)')
    print('  - MixUp Augmentation: ENABLED (30% probability, alpha=0.2)')
    print('  - Data Augmentation: ENABLED')
    print('    * ColorJitter: brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05')
    print('    * RandomErasing: probability=0.3, sl=0.02, sh=0.3, r1=0.3')
    print('  - Weight Initialization: ENABLED (Xavier uniform for LSTM/Linear)')
    print(f'  - Batch Size: {bs} (increased from 22, learning rate: 0.002)')
    print(f'  - Data Loading: {n_cpu} workers, pin_memory=True, prefetch_factor=4, persistent_workers=True')
    print('  - Model Compilation: See message above')
    print('='*60)
    print(f'Max iterations: {iterations}')
    print(f'Validation every {it_eval} iterations')
    print(f'Early stopping patience: {early_stop_patience} validation checks')
    print(f'Early stopping min delta: {early_stop_min_delta}')
    print(f'Budget limit: ${MAX_BUDGET} (Modal timeout: 9.5 hours)')
    print('='*60)

    i = 0
    mixup_prob = 0.3  # 30% chance of using MixUp
    while i < iterations:
        for batch_idx, sample in enumerate(data_loader):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            labels_flat = labels.view(bs*seq_length)
            
            # Mixed precision: Forward pass uses FP16 for speed
            with autocast(device_type='cuda'):
                # MixUp augmentation (30% probability)
                use_mixup = np.random.rand() < mixup_prob
                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_data(images, labels_flat, alpha=0.2)
                    labels_a = labels_a.view(bs*seq_length)
                    labels_b = labels_b.view(bs*seq_length)
                
                logits = model(images)
                
                if use_mixup:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = criterion(logits, labels_flat)
            
            # Gradient accumulation: scale loss
            loss = loss / accumulation_steps
            
            # Mixed precision: Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every N accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(base_optimizer)  # Use base optimizer for unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(base_optimizer)  # Step base optimizer
            scaler.update()
                optimizer.step()  # Step lookahead optimizer
                optimizer.zero_grad()
                # Learning rate scheduling
                scheduler.step()
                # Update EMA
                ema.update()
                # Update SWA (if past start point)
                if i >= swa_start and i % swa.swa_freq == 0:
                    swa.update_swa()
            
            # Use unscaled loss for logging (multiply back)
            losses.update(loss.item() * accumulation_steps, images.size(0))
            # Log cost estimate every 100 iterations
            if i % 100 == 0:
                elapsed_hours = (time.time() - start_time) / 3600
                estimated_cost = elapsed_hours * ESTIMATED_COST_PER_HOUR
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})\tTime: {:.1f}h\tEst. Cost: ${:.2f}'.format(
                    i, elapsed_hours, estimated_cost, loss=losses))
            else:
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            
            # Periodic checkpoint save
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict(),
                            'scaler_state_dict': scaler.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            
            # Periodic validation
            if i % it_eval == 0 and i > 0:
                print('\n' + '='*60)
                print(f'Running validation at iteration {i}...')
                try:
                    # Apply EMA weights for validation
                    ema.apply_shadow()
                    model.eval()
                    with torch.no_grad():
                        pce = eval(model, split, seq_length, n_cpu, disp=False, vid_dir='/data/videos_160/')
                    model.train()
                    ema.restore()  # Restore original weights for training
                    
                    print(f'Validation PCE: {pce:.4f} (Best: {best_pce:.4f})')
                    
                    # Check for improvement
                    if pce > best_pce + early_stop_min_delta:
                        best_pce = pce
                        best_iteration = i
                        no_improve_count = 0
                        # Apply EMA and save best model
                        ema.apply_shadow()
                        torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                    'base_optimizer_state_dict': base_optimizer.state_dict(),
                                    'model_state_dict': model.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'base_scheduler_state_dict': base_scheduler.state_dict()}, 
                                  'models/swingnet_best.pth.tar')
                        ema.restore()
                        print(f'*** New best model saved! (PCE: {best_pce:.4f}) ***')
                    else:
                        no_improve_count += 1
                        print(f'No improvement for {no_improve_count}/{early_stop_patience} validation checks')
                    
                    print('='*60 + '\n')
                    
                    # Early stopping check
                    if no_improve_count >= early_stop_patience:
                        elapsed_hours = (time.time() - start_time) / 3600
                        estimated_cost = elapsed_hours * ESTIMATED_COST_PER_HOUR
                        print(f'\n{"="*60}')
                        print(f'Early stopping triggered!')
                        print(f'No improvement for {early_stop_patience} consecutive validation checks')
                        print(f'Best PCE: {best_pce:.4f} at iteration {best_iteration}')
                        print(f'Total time: {elapsed_hours:.2f} hours')
                        print(f'Estimated cost: ${estimated_cost:.2f}')
                        print(f'{"="*60}\n')
                        break
                        
                except Exception as e:
                    print(f'ERROR during validation: {e}')
                    import traceback
                    traceback.print_exc()
                    print('Continuing training despite validation error...')
                    model.train()  # Make sure model is back in train mode
                    print('='*60 + '\n')
            
            if i == iterations:
                break

        # Break from outer loop if early stopping triggered
        if no_improve_count >= early_stop_patience:
            break

    elapsed_hours = (time.time() - start_time) / 3600
    estimated_cost = elapsed_hours * ESTIMATED_COST_PER_HOUR
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best PCE: {best_pce:.4f} at iteration {best_iteration}')
    print(f'Final iteration: {i}')
    print(f'Total time: {elapsed_hours:.2f} hours')
    print(f'Estimated cost: ${estimated_cost:.2f}')
    
    # Apply EMA weights to final model
    ema.apply_shadow()
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'base_optimizer_state_dict': base_optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(final_checkpoint, 'models/swingnet_final_ema.pth.tar')
    print('Best model (EMA) saved as: models/swingnet_best.pth.tar')
    print('Final model (EMA) saved as: models/swingnet_final_ema.pth.tar')
    
    # Apply SWA if it was used
    if swa.swa_model is not None:
        swa.apply_swa()
        torch.save({'model_state_dict': model.state_dict()}, 
                  'models/swingnet_final_swa.pth.tar')
        print('Final model (SWA) saved as: models/swingnet_final_swa.pth.tar')
        ema.restore()  # Restore EMA weights
    
    print('='*60)
    
    # Write cost to file so modal_runner can read it
    with open('training_cost.txt', 'w') as f:
        f.write(f'{estimated_cost:.2f}')