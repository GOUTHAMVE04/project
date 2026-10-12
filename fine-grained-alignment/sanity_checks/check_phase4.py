"""
Sanity check script for Phase 4: Loss & Beta Schedule.
Verifies PatchAlignmentLoss (Eq. 5), beta scheduling, and manual sanity checks:
- Identity test: feeding label's own embedding yields L_patch ≈ 0.
- Orthogonal test: L_patch ≈ 1.0.
- Opposite test: L_patch ≈ 2.0.
- Real batch loss computation and backward pass on MPS.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from losses.patch_alignment_loss import PatchAlignmentLoss, BetaScheduler
from model.word_embeddings import FrozenWordEmbeddings
from model.projector import PatchProjector

def run_phase4_checks():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Phase 4 Sanity Checks on Device: {device} ===\n")

    criterion = PatchAlignmentLoss()

    # -------------------------------------------------------------
    # Test 1: Manual Sanity Check — Perfect Alignment (Loss ≈ 0)
    # -------------------------------------------------------------
    print("--- 1. Manual Sanity Check: Identical Embeddings (Identity Test) ---")
    d = 768
    num_patches = 49

    # Generate random normalized label vectors for 2 instances
    w1 = F.normalize(torch.randn(d, device=device), dim=-1)
    w2 = F.normalize(torch.randn(d, device=device), dim=-1)

    # Construct patch tensors where active mask patches are set identically to target embedding
    mask1 = torch.zeros(7, 7, device=device)
    mask1[2:5, 2:5] = 1.0  # 9 active patches
    
    mask2 = torch.zeros(7, 7, device=device)
    mask2[0:2, 0:2] = 1.0  # 4 active patches

    # Construct image patches: fill active patches with the exact target word embedding
    img1_patches = torch.zeros(num_patches, d, device=device)
    flat_mask1 = mask1.view(-1).bool()
    img1_patches[flat_mask1] = w1  # Perfect match

    img2_patches = torch.zeros(num_patches, d, device=device)
    flat_mask2 = mask2.view(-1).bool()
    img2_patches[flat_mask2] = w2  # Perfect match

    batch_patches = torch.stack([img1_patches, img2_patches], dim=0)  # (2, 49, 768)
    batch_instances = [
        [{"patch_mask": mask1}],
        [{"patch_mask": mask2}]
    ]
    target_embeds = [
        [w1],
        [w2]
    ]

    loss_identical, count = criterion(batch_patches, batch_instances, target_embeds)
    print(f"Number of instances evaluated: {count}")
    print(f"Loss with identical embeddings: {loss_identical.item():.8f}")
    assert loss_identical.item() < 1e-5, f"Expected loss ≈ 0, got {loss_identical.item()}"
    print("✓ PASSED: Feeding target label's own embedding yields L_patch ≈ 0.0.")

    # -------------------------------------------------------------
    # Test 2: Orthogonal & Anti-Parallel Embedding Verification
    # -------------------------------------------------------------
    print("\n--- 2. Geometric Limit Tests: Orthogonal and Opposite Embeddings ---")
    # Opposite embedding: patch = -w1 -> cosine_similarity = -1.0 -> loss = 2.0
    img_opp_patches = torch.zeros(num_patches, d, device=device)
    img_opp_patches[flat_mask1] = -w1
    loss_opposite, _ = criterion(img_opp_patches.unsqueeze(0), [[{"patch_mask": mask1}]], [[w1]])
    print(f"Loss with opposite embeddings (cos=-1.0):   {loss_opposite.item():.6f} (Expected: 2.0)")
    assert abs(loss_opposite.item() - 2.0) < 1e-5, "Opposite embedding loss must equal 2.0"

    # Orthogonal embedding: patch vector orthogonal to w1 -> cosine_similarity = 0.0 -> loss = 1.0
    w_rand = torch.randn(d, device=device)
    w_ortho = w_rand - (w_rand @ w1) * w1
    w_ortho = F.normalize(w_ortho, dim=-1)
    img_ortho_patches = torch.zeros(num_patches, d, device=device)
    img_ortho_patches[flat_mask1] = w_ortho
    loss_ortho, _ = criterion(img_ortho_patches.unsqueeze(0), [[{"patch_mask": mask1}]], [[w1]])
    print(f"Loss with orthogonal embeddings (cos=0.0): {loss_ortho.item():.6f} (Expected: 1.0)")
    assert abs(loss_ortho.item() - 1.0) < 1e-4, "Orthogonal embedding loss must equal 1.0"
    print("✓ PASSED: Geometric alignment bounds verified correctly.")

    # -------------------------------------------------------------
    # Test 3: Beta Scheduler Verification
    # -------------------------------------------------------------
    print("\n--- 3. Testing Beta Schedule (0 → 5 linear ramp) ---")
    total_steps = 200
    scheduler = BetaScheduler(beta_start=0.0, beta_end=5.0, total_steps=total_steps)

    b_0 = scheduler.get_beta(0)
    b_mid = scheduler.get_beta(100)
    b_end = scheduler.get_beta(200)
    b_past = scheduler.get_beta(300)

    print(f"Beta at step 0:   {b_0:.2f} (Expected: 0.0)")
    print(f"Beta at step 100: {b_mid:.2f} (Expected: 2.5)")
    print(f"Beta at step 200: {b_end:.2f} (Expected: 5.0)")
    print(f"Beta at step 300: {b_past:.2f} (Expected: 5.0 capped)")

    assert abs(b_0 - 0.0) < 1e-6
    assert abs(b_mid - 2.5) < 1e-6
    assert abs(b_end - 5.0) < 1e-6
    assert abs(b_past - 5.0) < 1e-6
    print("✓ PASSED: Linear Beta schedule strictly adheres to ablation specifications.")

    # -------------------------------------------------------------
    # Test 4: End-to-End Batch Forward & Backward Pass
    # -------------------------------------------------------------
    print("\n--- 4. End-to-End Batch Loss with Trainable Projector ---")
    projector = PatchProjector(in_dim=768, hidden_dim=768, out_dim=768).to(device)
    word_enc = FrozenWordEmbeddings(device=device)

    # Random visual patch tokens simulating CLIP output
    raw_patches = torch.randn(2, 49, 768, device=device)
    projected = projector(raw_patches)

    test_instances = [
        [
            {"patch_mask": mask1, "category_name": "cat"},
            {"patch_mask": mask2, "category_name": "dog"}
        ],
        [
            {"patch_mask": mask1, "category_name": "car"}
        ]
    ]

    target_embeddings = []
    for img_insts in test_instances:
        cat_names = [inst["category_name"] for inst in img_insts]
        embeds = word_enc.encode_labels(cat_names).to(device)
        target_embeddings.append([embeds[i] for i in range(len(cat_names))])

    loss, total_instances = criterion(projected, test_instances, target_embeddings)
    beta = scheduler.get_beta(50)
    total_loss = beta * loss

    print(f"Evaluated {total_instances} instances across 2 images.")
    print(f"Raw L_patch:    {loss.item():.4f}")
    print(f"Beta (step 50): {beta:.2f}")
    print(f"Scaled Loss:    {total_loss.item():.4f}")

    projector.zero_grad()
    total_loss.backward()

    has_grads = all(p.grad is not None and p.grad.norm().item() > 0 for p in projector.parameters())
    print(f"Projector parameters received valid gradients: {has_grads}")
    assert has_grads, "Gradients failed to propagate to projector!"

    print("\n=== Phase 4 Sanity Checks PASSED Successfully! ===")

if __name__ == "__main__":
    run_phase4_checks()
