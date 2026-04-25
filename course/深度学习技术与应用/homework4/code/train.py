"""
Train and Evaluate Seq2Seq Transformer for Code Generation
"""
import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from dataset import ConcodeDataset, build_vocab, collate_fn, Vocabulary
from model import Seq2SeqTransformer, count_parameters


def set_device():
    """Set device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_epoch(model, dataloader, optimizer, criterion, device, tgt_vocab):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, (src, tgt, src_mask) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        
        # Prepare input and labels
        # tgt[:, :-1] as input, tgt[:, 1:] as labels (teacher forcing)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        
        # Forward
        logits = model(src, tgt_input, src_mask)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device, tgt_vocab):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for src, tgt, src_mask in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            
            logits = model(src, tgt_input, src_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def beam_search_decode(model, src, src_mask, device, tgt_vocab, max_len=50, beam_size=3):
    """
    Beam search decoding.
    
    Args:
        model: Trained model
        src: (1, src_len) Source tokens
        src_mask: (1, src_len) Source mask
        beam_size: Beam size
    
    Returns:
        decoded_tokens: List of tokens
    """
    model.eval()
    bos_idx = tgt_vocab.BOS_TOKEN_IDX
    eos_idx = tgt_vocab.EOS_TOKEN_IDX
    
    # Encode source
    memory = model.encode(src.to(device), src_mask.to(device))
    
    # Initialize beam
    beams = [(torch.tensor([[bos_idx]], device=device), 0.0)]
    completed = []
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed.append((seq, score))
                continue
            
            # Decode one step
            tgt_mask = model.generate_square_subsequent_mask(seq.size(1)).to(device)
            logits = model.decode(seq, memory, tgt_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            
            # Get top-k candidates
            topk_log_probs, topk_indices = log_probs.topk(beam_size)
            
            for i in range(beam_size):
                token = topk_indices[0, i].item()
                new_score = score + topk_log_probs[0, i].item()
                new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=1)
                all_candidates.append((new_seq, new_score))
        
        if not all_candidates:
            break
        
        # Select top beams
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]
        
        if len(completed) >= beam_size:
            break
    
    # Select best sequence
    if completed:
        all_seqs = beams + completed
    else:
        all_seqs = beams
    
    all_seqs.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)  # Length-normalized score
    best_seq = all_seqs[0][0][0]
    
    # Convert to tokens
    tokens = []
    for idx in best_seq:
        idx = idx.item()
        if idx == eos_idx:
            break
        if idx not in [bos_idx, tgt_vocab.PAD_TOKEN_IDX]:
            tokens.append(tgt_vocab.idx2token.get(idx, tgt_vocab.UNK_TOKEN))
    
    return tokens


def greedy_decode(model, src, src_mask, device, tgt_vocab, max_len=50):
    """Greedy decoding (faster but less accurate)."""
    model.eval()
    bos_idx = tgt_vocab.BOS_TOKEN_IDX
    eos_idx = tgt_vocab.EOS_TOKEN_IDX
    
    memory = model.encode(src.to(device), src_mask.to(device))
    
    seq = torch.tensor([[bos_idx]], device=device)
    
    for _ in range(max_len):
        tgt_mask = model.generate_square_subsequent_mask(seq.size(1)).to(device)
        logits = model.decode(seq, memory, tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == eos_idx:
            break
        
        seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
    
    # Convert to tokens
    tokens = []
    for idx in seq[0]:
        idx = idx.item()
        if idx == eos_idx:
            break
        if idx not in [bos_idx, tgt_vocab.PAD_TOKEN_IDX]:
            tokens.append(tgt_vocab.idx2token.get(idx, tgt_vocab.UNK_TOKEN))
    
    return tokens


def compute_bleu(reference, hypothesis):
    """Compute BLEU score for a single pair."""
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split() if isinstance(hypothesis, str) else hypothesis
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Compute precisions
    precisions = []
    for n in range(1, 5):
        if len(hyp_tokens) >= n:
            hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            if total > 0:
                precisions.append(overlap / total)
            else:
                precisions.append(0)
        else:
            precisions.append(0)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len)
    
    return 100 * geo_mean * bp


def evaluate_predictions(predictions, references):
    """Compute Exact Match and BLEU scores."""
    exact_matches = 0
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Exact Match
        if pred_tokens == ref_tokens:
            exact_matches += 1
        
        # BLEU
        bleu = compute_bleu(ref, pred)
        bleu_scores.append(bleu)
    
    em = (exact_matches / len(predictions)) * 100 if predictions else 0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    
    return em, avg_bleu


def train(args):
    """Main training function."""
    device = set_device()
    print(f"Using device: {device}")

    # Data paths
    train_path = os.path.join(args.data_dir, 'train.jsonl')
    dev_path = os.path.join(args.data_dir, 'dev.jsonl')
    test_path = os.path.join(args.data_dir, 'test.jsonl')
    
    # Build vocabulary
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    if os.path.exists(vocab_path) and not args.overwrite_cache:
        print("Loading cached vocabulary...")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            src_vocab = Vocabulary()
            tgt_vocab = Vocabulary()
            src_vocab.token2idx = vocab_data['src_token2idx']
            src_vocab.idx2token = {int(k): v for k, v in vocab_data['src_idx2token'].items()}
            src_vocab.n_tokens = len(src_vocab.token2idx)
            tgt_vocab.token2idx = vocab_data['tgt_token2idx']
            tgt_vocab.idx2token = {int(k): v for k, v in vocab_data['tgt_idx2token'].items()}
            tgt_vocab.n_tokens = len(tgt_vocab.token2idx)
    else:
        print("Building vocabulary...")
        src_vocab, tgt_vocab = build_vocab([train_path, dev_path], min_freq=2)
        
        # Save vocabulary
        vocab_data = {
            'src_token2idx': src_vocab.token2idx,
            'src_idx2token': {str(k): v for k, v in src_vocab.idx2token.items()},
            'tgt_token2idx': tgt_vocab.token2idx,
            'tgt_idx2token': {str(k): v for k, v in tgt_vocab.idx2token.items()},
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ConcodeDataset(train_path, src_vocab, tgt_vocab, max_len=args.max_len, mode='train')
    dev_dataset = ConcodeDataset(dev_path, src_vocab, tgt_vocab, max_len=args.max_len, mode='dev')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    # Create model
    print("Creating model...")
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_TOKEN_IDX)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tgt_vocab)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, dev_loader, criterion, device, tgt_vocab)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'src_vocab': src_vocab.token2idx,
                'tgt_vocab': tgt_vocab.token2idx,
            }, model_path)
            print(f"  Saved best model to {model_path}")
    
    # Save training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'))
    print(f"\nTraining curve saved to {os.path.join(args.output_dir, 'training_curve.png')}")
    
    # Save learning curve data
    with open(os.path.join(args.output_dir, 'learning_curve.json'), 'w') as f:
        json.dump({'train_loss': train_losses, 'val_loss': val_losses}, f)
    
    print("\nTraining completed!")
    
    return model, src_vocab, tgt_vocab, train_losses, val_losses


def generate_predictions(args):
    """Generate predictions on test set."""
    device = set_device()
    
    # Load model
    model_path = os.path.join(args.output_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.token2idx = checkpoint['src_vocab']
    src_vocab.idx2token = {int(k): v for k, v in checkpoint['src_vocab'].items()}
    src_vocab.n_tokens = len(src_vocab.token2idx)
    tgt_vocab.token2idx = checkpoint['tgt_vocab']
    tgt_vocab.idx2token = {int(k): v for k, v in checkpoint['tgt_vocab'].items()}
    tgt_vocab.n_tokens = len(tgt_vocab.token2idx)
    
    # Create model
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Load test dataset
    test_path = os.path.join(args.data_dir, 'test.jsonl')
    test_dataset = ConcodeDataset(test_path, src_vocab, tgt_vocab, max_len=args.max_len, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print(f"Generating predictions for {len(test_dataset)} samples...")
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for idx, (src, tgt, src_mask) in enumerate(test_loader):
            # Generate prediction
            if args.decode == 'greedy':
                tokens = greedy_decode(model, src, src_mask, device, tgt_vocab, args.max_len)
            else:
                tokens = beam_search_decode(model, src, src_mask, device, tgt_vocab, args.max_len, args.beam_size)
            
            pred_code = ' '.join(tokens)
            predictions.append(pred_code)
            
            # Get reference
            ref_code = test_dataset.get_raw_code(idx)
            references.append(ref_code)
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(test_dataset)} samples")
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, 'predictions.txt')
    with open(pred_path, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"Predictions saved to {pred_path}")
    
    # Save references in required format
    ref_path = os.path.join(args.output_dir, 'references.json')
    with open(ref_path, 'w') as f:
        for ref in references:
            f.write(json.dumps({'code': ref}) + '\n')
    print(f"References saved to {ref_path}")
    
    # Evaluate
    em, bleu = evaluate_predictions(predictions, references)
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Exact Match: {em:.2f}%")
    print(f"BLEU Score: {bleu:.2f}")
    print("="*50)
    
    # Save results
    results = {'exact_match': em, 'bleu': bleu, 'n_samples': len(predictions)}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return em, bleu


def main():
    parser = argparse.ArgumentParser(description='Train and Evaluate Code Generation Model')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    
    # Decoding
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'], help='Decoding method')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search')
    
    # Other
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'both'], help='Mode')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite cached data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ['train', 'both']:
        train_losses, val_losses = train(args)
    
    if args.mode in ['predict', 'both']:
        em, bleu = generate_predictions(args)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
