"""
Adaptive Parameter Selection Methods for Hybrid DynaPrompt

This module implements two methods for dynamically selecting optimal feedback parameters:
- Method 1: Baseline Quality Assessment + Decision Rules (fast, rule-based)
- Method 4: Meta-Learning Predictor (data-driven, generalizable)

Both methods address the core problem: fixed parameters (alpha=0.07, boost=1.3) are
optimal for weak baselines but too aggressive for strong baselines, causing over-optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json
from pathlib import Path


# ============================================================================
# METHOD 1: Baseline Quality Assessment + Decision Rules
# ============================================================================

class BaselineQualityAssessor:
    """
    Assess baseline generation quality in first 10 steps using CLIP score.
    Use decision rules to select appropriate alpha and boost_factor.
    
    Rationale:
    - Weak baselines (CLIP < 35): Need strong feedback (alpha=0.07-0.10)
    - Medium baselines (CLIP 35-55): Need moderate feedback (alpha=0.04-0.07)
    - Strong baselines (CLIP 55-70): Need gentle feedback (alpha=0.02-0.04)
    - Very strong baselines (CLIP > 70): Minimal/no feedback (alpha=0.00-0.02)
    """
    
    def __init__(self, clip_model, device='cuda'):
        """
        Initialize baseline quality assessor
        
        Args:
            clip_model: CLIP model for scoring
            device: Torch device
        """
        self.clip_model = clip_model
        self.device = device
        
        # Quality tier thresholds (CLIP scores)
        self.tiers = {
            'very_weak': (0, 35),
            'weak': (35, 45),
            'medium': (45, 55),
            'strong': (55, 65),
            'very_strong': (65, 100)
        }
        
        # Parameter recommendations per tier
        self.param_rules = {
            'very_weak': {'alpha': 0.10, 'boost_factor': 1.5, 'frequency': 3},
            'weak': {'alpha': 0.07, 'boost_factor': 1.3, 'frequency': 4},
            'medium': {'alpha': 0.05, 'boost_factor': 1.2, 'frequency': 5},
            'strong': {'alpha': 0.03, 'boost_factor': 1.1, 'frequency': 6},
            'very_strong': {'alpha': 0.01, 'boost_factor': 1.05, 'frequency': 8}
        }
    
    def assess_baseline_quality(
        self,
        baseline_model,
        prompt: str,
        num_assessment_steps: int = 10,
        seed: int = 42
    ) -> Dict:
        """
        Generate baseline image for first N steps and measure CLIP score
        
        Args:
            baseline_model: Stable Diffusion model (no feedback)
            prompt: Text prompt
            num_assessment_steps: Number of steps to run (default 10)
            seed: Random seed for reproducibility
            
        Returns:
            Dict with keys: clip_score, quality_tier, recommended_params
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate partial baseline image (first N steps)
        with torch.no_grad():
            # Initialize latent noise
            latent = torch.randn(1, 4, 64, 64, device=self.device)
            
            # Encode prompt
            text_input = baseline_model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=baseline_model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = baseline_model.text_encoder(text_input.input_ids.to(self.device))[0]
            
            # Run partial denoising (first N steps)
            for step in range(num_assessment_steps):
                # Predict noise
                noise_pred = baseline_model.unet(
                    latent,
                    step,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Denoise step
                latent = baseline_model.scheduler.step(noise_pred, step, latent).prev_sample
            
            # Decode latent to image
            image = baseline_model.vae.decode(latent / 0.18215).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            
            # Compute CLIP score
            clip_score = self._compute_clip_score(image, prompt)
        
        # Determine quality tier
        quality_tier = self._classify_quality_tier(clip_score)
        
        # Get recommended parameters
        recommended_params = self.param_rules[quality_tier]
        
        return {
            'clip_score': float(clip_score),
            'quality_tier': quality_tier,
            'recommended_params': recommended_params,
            'assessment_steps': num_assessment_steps
        }
    
    def _compute_clip_score(self, image: np.ndarray, prompt: str) -> float:
        """
        Compute CLIP score between image and prompt
        
        Args:
            image: RGB image (H, W, 3) in range [0, 1]
            prompt: Text prompt
            
        Returns:
            CLIP score (0-100 scale)
        """
        from PIL import Image
        import clip
        
        # Convert numpy to PIL
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Preprocess image
        image_input = clip.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Encode image and text
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (cosine similarity -> 0-100 scale)
            similarity = (image_features @ text_features.T).item()
            clip_score = (similarity + 1) * 50  # Map [-1, 1] to [0, 100]
        
        return clip_score
    
    def _classify_quality_tier(self, clip_score: float) -> str:
        """
        Classify CLIP score into quality tier
        
        Args:
            clip_score: CLIP score (0-100)
            
        Returns:
            Quality tier name
        """
        for tier_name, (low, high) in self.tiers.items():
            if low <= clip_score < high:
                return tier_name
        
        # Default to very_strong if score > 100 (shouldn't happen)
        return 'very_strong'
    
    def select_adaptive_parameters(
        self,
        baseline_model,
        prompt: str,
        verbose: bool = True
    ) -> Dict:
        """
        Main API: Assess baseline and return recommended parameters
        
        Args:
            baseline_model: SD model for baseline assessment
            prompt: Text prompt
            verbose: Print assessment results
            
        Returns:
            Dict with recommended alpha, boost_factor, frequency
        """
        assessment = self.assess_baseline_quality(baseline_model, prompt)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Baseline Quality Assessment")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"CLIP Score (10 steps): {assessment['clip_score']:.2f}")
            print(f"Quality Tier: {assessment['quality_tier']}")
            print(f"\nRecommended Parameters:")
            print(f"  alpha:        {assessment['recommended_params']['alpha']:.3f}")
            print(f"  boost_factor: {assessment['recommended_params']['boost_factor']:.2f}")
            print(f"  frequency:    {assessment['recommended_params']['frequency']}")
            print(f"{'='*60}\n")
        
        return assessment['recommended_params']


# ============================================================================
# METHOD 4: Meta-Learning Predictor
# ============================================================================

class ParameterPredictorMLP(nn.Module):
    """
    Neural network that predicts optimal (alpha, boost_factor) from prompt embedding
    
    Architecture:
    - Input: CLIP text embedding (512-dim) + baseline CLIP score (1-dim)
    - Hidden: 256 -> 128 -> 64
    - Output: alpha (1), boost_factor (1), frequency (1)
    
    Training:
    - Dataset: (prompt_embedding, baseline_clip) -> (optimal_alpha, optimal_boost, optimal_freq)
    - Loss: MSE on parameter predictions
    - Validation: Measure CLIP improvement on held-out prompts
    """
    
    def __init__(self, input_dim: int = 513, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize parameter predictor network
        
        Args:
            input_dim: Input dimension (512 CLIP + 1 baseline score = 513)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (3 parameters)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Output activation to constrain parameter ranges
        self.alpha_activation = nn.Sigmoid()  # Map to [0, 1], then scale to [0, 0.15]
        self.boost_activation = nn.Sigmoid()  # Map to [0, 1], then scale to [1.0, 2.0]
        self.freq_activation = nn.Sigmoid()   # Map to [0, 1], then scale to [2, 10]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict parameters from input features
        
        Args:
            x: Input tensor (batch, 513) - [CLIP embedding (512), baseline score (1)]
            
        Returns:
            Dict with alpha, boost_factor, frequency predictions
        """
        # Network output
        raw_output = self.network(x)
        
        # Apply constrained activations
        alpha = self.alpha_activation(raw_output[:, 0]) * 0.15  # [0, 0.15]
        boost = 1.0 + self.boost_activation(raw_output[:, 1]) * 1.0  # [1.0, 2.0]
        freq = 2 + self.freq_activation(raw_output[:, 2]) * 8  # [2, 10]
        
        return {
            'alpha': alpha,
            'boost_factor': boost,
            'frequency': freq
        }


class MetaLearningPredictor:
    """
    Meta-learning system for predicting optimal parameters
    
    Training procedure:
    1. Collect dataset: For N prompts, sweep parameters and find optimal values
    2. Extract features: CLIP text embedding + baseline CLIP score
    3. Train MLP: Map features -> optimal parameters
    4. Evaluate: Test on held-out prompts, measure CLIP improvement
    """
    
    def __init__(self, clip_model, device='cuda'):
        """
        Initialize meta-learning predictor
        
        Args:
            clip_model: CLIP model for feature extraction
            device: Torch device
        """
        self.clip_model = clip_model
        self.device = device
        
        # Initialize predictor network
        self.predictor = ParameterPredictorMLP().to(device)
        
        # Training state
        self.is_trained = False
        self.train_history = []
    
    def extract_features(
        self,
        prompt: str,
        baseline_clip_score: float
    ) -> torch.Tensor:
        """
        Extract features for predictor input
        
        Args:
            prompt: Text prompt
            baseline_clip_score: CLIP score of baseline generation
            
        Returns:
            Feature tensor (513-dim): [CLIP embedding (512), baseline score (1)]
        """
        import clip
        
        # Encode prompt with CLIP
        text_input = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Concatenate with baseline score
        baseline_tensor = torch.tensor([baseline_clip_score / 100.0], device=self.device)
        features = torch.cat([text_features.squeeze(0), baseline_tensor])
        
        return features
    
    def collect_training_data(
        self,
        baseline_model,
        hybrid_model,
        prompts: List[str],
        alpha_range: Tuple[float, float] = (0.01, 0.15),
        boost_range: Tuple[float, float] = (1.0, 2.0),
        num_samples_per_prompt: int = 10
    ) -> List[Dict]:
        """
        Collect training dataset by sweeping parameters for each prompt
        
        Args:
            baseline_model: SD model for baseline generation
            hybrid_model: Hybrid model for parameter sweeping
            prompts: List of prompts to evaluate
            alpha_range: Alpha sweep range
            boost_range: Boost factor sweep range
            num_samples_per_prompt: Number of parameter combinations to try
            
        Returns:
            List of training examples: [{features, optimal_params}, ...]
        """
        training_data = []
        
        print(f"Collecting training data for {len(prompts)} prompts...")
        print(f"Sampling {num_samples_per_prompt} parameter combinations per prompt")
        
        for prompt in prompts:
            print(f"\nProcessing: {prompt}")
            
            # 1. Generate baseline
            baseline_result = baseline_model.generate(prompt, num_steps=50)
            baseline_clip = baseline_result['clip_score']
            
            # 2. Sweep parameters
            best_clip = baseline_clip
            best_params = {'alpha': 0.0, 'boost_factor': 1.0, 'frequency': 4}
            
            for _ in range(num_samples_per_prompt):
                # Sample random parameters
                alpha = np.random.uniform(*alpha_range)
                boost = np.random.uniform(*boost_range)
                freq = np.random.randint(2, 9)
                
                # Generate with these parameters
                hybrid_result = hybrid_model.generate(
                    prompt,
                    num_steps=50,
                    alpha=alpha,
                    boost_factor=boost,
                    feedback_frequency=freq
                )
                
                # Track best
                if hybrid_result['clip_score'] > best_clip:
                    best_clip = hybrid_result['clip_score']
                    best_params = {
                        'alpha': alpha,
                        'boost_factor': boost,
                        'frequency': freq
                    }
            
            # 3. Extract features and store
            features = self.extract_features(prompt, baseline_clip)
            
            training_data.append({
                'prompt': prompt,
                'features': features.cpu().numpy(),
                'baseline_clip': baseline_clip,
                'optimal_params': best_params,
                'optimal_clip': best_clip,
                'improvement': best_clip - baseline_clip
            })
            
            print(f"  Baseline CLIP: {baseline_clip:.2f}")
            print(f"  Optimal CLIP: {best_clip:.2f} (+{best_clip - baseline_clip:.2f})")
            print(f"  Optimal params: alpha={best_params['alpha']:.3f}, "
                  f"boost={best_params['boost_factor']:.2f}, freq={best_params['frequency']}")
        
        return training_data
    
    def train(
        self,
        training_data: List[Dict],
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        validation_split: float = 0.2
    ):
        """
        Train parameter predictor network
        
        Args:
            training_data: List of training examples from collect_training_data()
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            validation_split: Fraction of data for validation
        """
        print(f"\nTraining parameter predictor...")
        print(f"Dataset size: {len(training_data)}")
        print(f"Epochs: {num_epochs}, LR: {learning_rate}, Batch size: {batch_size}")
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Prepare tensors
        train_features = torch.stack([
            torch.from_numpy(ex['features']).float() for ex in train_data
        ]).to(self.device)
        
        train_targets = torch.tensor([
            [ex['optimal_params']['alpha'],
             ex['optimal_params']['boost_factor'],
             ex['optimal_params']['frequency']]
            for ex in train_data
        ], dtype=torch.float32).to(self.device)
        
        val_features = torch.stack([
            torch.from_numpy(ex['features']).float() for ex in val_data
        ]).to(self.device)
        
        val_targets = torch.tensor([
            [ex['optimal_params']['alpha'],
             ex['optimal_params']['boost_factor'],
             ex['optimal_params']['frequency']]
            for ex in val_data
        ], dtype=torch.float32).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            self.predictor.train()
            
            # Mini-batch training
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                # Forward
                predictions = self.predictor(batch_features)
                pred_tensor = torch.stack([
                    predictions['alpha'],
                    predictions['boost_factor'],
                    predictions['frequency']
                ], dim=1)
                
                # Loss
                loss = criterion(pred_tensor, batch_targets)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            if (epoch + 1) % 10 == 0:
                self.predictor.eval()
                with torch.no_grad():
                    val_predictions = self.predictor(val_features)
                    val_pred_tensor = torch.stack([
                        val_predictions['alpha'],
                        val_predictions['boost_factor'],
                        val_predictions['frequency']
                    ], dim=1)
                    val_loss = criterion(val_pred_tensor, val_targets)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                
                self.train_history.append({
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'val_loss': val_loss.item()
                })
        
        self.is_trained = True
        print("Training complete!")
    
    def predict_parameters(self, prompt: str, baseline_clip_score: float) -> Dict:
        """
        Predict optimal parameters for a new prompt
        
        Args:
            prompt: Text prompt
            baseline_clip_score: Baseline CLIP score (from 10-step assessment)
            
        Returns:
            Dict with predicted alpha, boost_factor, frequency
        """
        if not self.is_trained:
            raise RuntimeError("Predictor not trained! Call train() first.")
        
        # Extract features
        features = self.extract_features(prompt, baseline_clip_score).unsqueeze(0)
        
        # Predict
        self.predictor.eval()
        with torch.no_grad():
            predictions = self.predictor(features)
        
        return {
            'alpha': predictions['alpha'].item(),
            'boost_factor': predictions['boost_factor'].item(),
            'frequency': int(predictions['frequency'].item())
        }
    
    def save(self, save_path: str):
        """Save trained predictor"""
        torch.save({
            'model_state_dict': self.predictor.state_dict(),
            'train_history': self.train_history
        }, save_path)
        print(f"Saved predictor to {save_path}")
    
    def load(self, load_path: str):
        """Load trained predictor"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint['train_history']
        self.is_trained = True
        print(f"Loaded predictor from {load_path}")


# ============================================================================
# Comparison & Evaluation
# ============================================================================

def compare_methods(
    baseline_model,
    hybrid_model,
    test_prompts: List[str],
    method1_assessor: BaselineQualityAssessor,
    method4_predictor: MetaLearningPredictor
) -> Dict:
    """
    Compare fixed parameters vs Method 1 vs Method 4 on test prompts
    
    Args:
        baseline_model: SD model for baseline
        hybrid_model: Hybrid model for generation
        test_prompts: List of test prompts
        method1_assessor: Trained Method 1 assessor
        method4_predictor: Trained Method 4 predictor
        
    Returns:
        Dict with comparison results
    """
    results = {
        'fixed': [],
        'method1': [],
        'method4': []
    }
    
    for prompt in test_prompts:
        print(f"\nEvaluating: {prompt}")
        
        # Baseline
        baseline_result = baseline_model.generate(prompt, num_steps=50)
        baseline_clip = baseline_result['clip_score']
        
        # Fixed parameters (alpha=0.07, boost=1.3, freq=4)
        fixed_result = hybrid_model.generate(
            prompt, num_steps=50,
            alpha=0.07, boost_factor=1.3, feedback_frequency=4
        )
        
        # Method 1: Adaptive rules
        method1_params = method1_assessor.select_adaptive_parameters(
            baseline_model, prompt, verbose=False
        )
        method1_result = hybrid_model.generate(
            prompt, num_steps=50, **method1_params
        )
        
        # Method 4: Meta-learning
        method4_params = method4_predictor.predict_parameters(prompt, baseline_clip)
        method4_result = hybrid_model.generate(
            prompt, num_steps=50, **method4_params
        )
        
        # Store results
        results['fixed'].append({
            'prompt': prompt,
            'baseline_clip': baseline_clip,
            'hybrid_clip': fixed_result['clip_score'],
            'improvement': fixed_result['clip_score'] - baseline_clip
        })
        
        results['method1'].append({
            'prompt': prompt,
            'baseline_clip': baseline_clip,
            'hybrid_clip': method1_result['clip_score'],
            'improvement': method1_result['clip_score'] - baseline_clip,
            'params': method1_params
        })
        
        results['method4'].append({
            'prompt': prompt,
            'baseline_clip': baseline_clip,
            'hybrid_clip': method4_result['clip_score'],
            'improvement': method4_result['clip_score'] - baseline_clip,
            'params': method4_params
        })
        
        print(f"  Baseline: {baseline_clip:.2f}")
        print(f"  Fixed:    {fixed_result['clip_score']:.2f} "
              f"(Δ{fixed_result['clip_score'] - baseline_clip:+.2f})")
        print(f"  Method 1: {method1_result['clip_score']:.2f} "
              f"(Δ{method1_result['clip_score'] - baseline_clip:+.2f})")
        print(f"  Method 4: {method4_result['clip_score']:.2f} "
              f"(Δ{method4_result['clip_score'] - baseline_clip:+.2f})")
    
    return results


if __name__ == '__main__':
    print("Adaptive Parameter Selection Methods")
    print("Method 1: Baseline Quality Assessment + Decision Rules")
    print("Method 4: Meta-Learning Predictor")
    print("\nUse these classes to implement adaptive parameter selection in your hybrid pipeline.")
