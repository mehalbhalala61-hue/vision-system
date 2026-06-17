# Debug Log — Vision System Capstone v3
> **Format:** Timestamp · Description · Hypothesis · Fix · Learning  
> **Purpose:** Documents every real bug encountered during development.  
> Interview answer for *"What challenges did you face?"*

---

## Bug #1 — Grad-CAM heatmaps blank after checkpoint reload
**Date:** Day 6 | **Severity:** High — silently produces wrong output

**Description:**  
After loading `best.pth` with `model.load_state_dict()`, Grad-CAM heatmaps were completely blank (all zeros). Model predictions were correct, only visualisation was broken.

**Hypothesis:**  
PyTorch's `torch.load()` reconstructs model weights but does NOT preserve forward hooks. Hooks registered before `load_state_dict()` silently detach — `_gradcam_features` list stays empty on forward pass.

**Reproduction:**
```python
model = build_model()
model._register_gradcam_hook()                    # hook registered
ckpt  = torch.load("best.pth")
model.load_state_dict(ckpt["model_state_dict"])   # hook silently detaches
model(dummy_input)
print(model._gradcam_features)                    # [] — empty!
```

**Fix:**  
Created `utils/checkpoint.py → load_checkpoint_with_hooks()` which calls `model._register_gradcam_hook()` immediately after `load_state_dict()`.

```python
def load_checkpoint_with_hooks(path, model, ...):
    model.load_state_dict(ckpt["model_state_dict"])
    model._register_gradcam_hook()   # re-register every time — v3 fix
```

**Learning:**  
PyTorch hooks are not part of `state_dict` — they live on the Python object, not serialised weights. Any function that loads a checkpoint must re-register hooks. Rule: *never use `model.load_state_dict()` directly — always use `load_checkpoint_with_hooks()`.*

---

## Bug #2 — NaN loss at epoch 3 with AMP
**Date:** Day 3 | **Severity:** Critical — training crashes

**Description:**  
Training ran fine for 2 epochs then loss became `nan` at epoch 3 batch 47. Happened consistently with same seed.

**Hypothesis:**  
AMP uses float16 for forward pass. float16 overflows at ~65504. With LR=1e-2 (guessed before LR finder), activations grew large enough to overflow in Bottleneck 1×1 expansion layer.

**Debugging steps:**
```python
# Added NaN check after each forward pass
if torch.isnan(loss):
    logger.error(f"NaN at batch {batch_idx}")
    # Traced: logits → layer3 output had inf values
    # Cause confirmed: LR was 10x too high
```

**Fix:**  
1. Ran LR finder → correct LR was `3e-4`, not `1e-2`  
2. `GradScaler` already present — handles float16 overflow  
3. Added gradient clipping `max_norm=1.0` as safety net  

**Learning:**  
AMP + high LR is the most common cause of NaN in ResNet training. Always run LR finder before training. `GradScaler` alone is not sufficient if LR is too high — it scales loss, not raw activations.

---

## Bug #3 — `postgres://` URL crashing SQLAlchemy on Railway
**Date:** Day 7B | **Severity:** High — production deploy broken

**Description:**  
Local dev worked perfectly. After deploying to Railway, FastAPI crashed immediately:  
`sqlalchemy.exc.NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:postgres`

**Hypothesis:**  
Railway injects `DATABASE_URL` as `postgres://` (legacy). SQLAlchemy 1.4+ dropped support — requires `postgresql://`.

**Fix:**
```python
# db/session.py — one line, applied once at startup
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)
```

**Learning:**  
Cloud platforms often inject legacy URL formats. Always normalise env vars at consumption point. The `.replace()` is a no-op if URL already uses `postgresql://` — safe for local dev.

---

## Bug #4 — `signal.alarm()` crash on Windows / Kaggle
**Date:** Day 7B | **Severity:** High — demo broken on non-Unix

**Description:**  
v2 used `signal.alarm(10)` for Gemini API timeout. Crashed on Windows:  
`AttributeError: module 'signal' has no attribute 'alarm'`

**Hypothesis:**  
`signal.alarm()` is Unix-only (POSIX). Windows does not implement it.

**Fix:**
```python
# Replaced signal.alarm() everywhere with:
response = await asyncio.wait_for(
    call_gemini(prompt),
    timeout=10.0
)
```
`asyncio.wait_for()` is cross-platform — Windows, Linux, macOS, Kaggle.

**Learning:**  
Never use Unix-specific APIs without an explicit OS check. `asyncio.wait_for()` is the correct cross-platform timeout for async operations and integrates cleanly with FastAPI's async handlers.

---

## Bug #5 — DataLoader `num_workers > 0` hang on Windows
**Date:** Day 1 | **Severity:** Medium — blocks local dev on Windows

**Description:**  
`DynamicDataset` with `num_workers=4` caused training script to hang indefinitely on Windows. No error, no output.

**Hypothesis:**  
Windows uses `spawn` for multiprocessing (not `fork`). DataLoader workers spawn new processes that try to re-import the main script → deadlock.

**Fix:**
```python
# dataset_loader.py
import platform
num_workers = cfg["dataloader"]["num_workers"]
if platform.system() == "Windows":
    num_workers = 0
```
Also added `if __name__ == "__main__": train()` guard in all scripts.

**Learning:**  
Always test DataLoader with `num_workers=0` first to isolate pipeline bugs from multiprocessing bugs. On Kaggle (Linux), `num_workers=4` gives ~30% speedup. Document Windows limitation clearly.

---

## Bug #6 — Nutrition CSV: zero calories for 12 classes after USDA fetch
**Date:** Day 1 | **Severity:** Medium — wrong data, interview risk

**Description:**  
After `fetch_nutrition.py`, 12 classes had `calories_per_100g = 0`. All were regional dishes (`pootharekulu`, `gavvalu`, `kajjikaya`) with no USDA record.

**Hypothesis:**  
`search_usda()` returned empty `foods` list → code wrote `0` to CSV instead of triggering fallback.

**Fix:**  
1. `search_usda()` returns `None` on empty results  
2. Explicit `if nutrients is None` → `get_fallback()` call  
3. Post-fetch zero-calorie validation with warning log  
4. `generate_usda_map.py` improves search terms via Gemini first

**Learning:**  
Always validate output data, not just input. A zero-calorie biryani is immediately caught by any interviewer. Data validation at write time is professional standard.

---

## Bug #7 — Wrong class labels silently (folder name mismatch)
**Date:** Day 1 | **Severity:** Medium — silent wrong labels

**Description:**  
10 epochs of training, 20% val accuracy (expected ~60%). No error.

**Hypothesis:**  
`classes.txt` had names like `aloo_gobi`. Kaggle folder names were `Aloo_Gobi`. Case mismatch → `class_to_idx` lookup failed → affected classes silently mapped to label 0.

**Fix:**
```python
# After dataset download, regenerate classes.txt from actual folders:
actual_classes = sorted(os.listdir("data/raw/train"))
with open("data/classes.txt", "w") as f:
    f.write("\n".join(actual_classes))
```
Added explicit warning in `DynamicDataset._scan_directory()` for any folder not in `classes.txt`.

**Learning:**  
Never assume folder names match expected class names. Always generate `classes.txt` from actual dataset structure. Make mismatches loud (warning log), not silent.

---

## Bug #8 — Soft label shape mismatch in overfit test
**Date:** Day 4 | **Severity:** Medium — overfit test broken after augmentation

**Description:**  
After integrating `AugmentationManager`, overfit test crashed:  
`RuntimeError: shapes [4, 101] vs [4]` — `SoftCrossEntropyLoss` expected `(B, C)` float but got `(B,)` int.

**Hypothesis:**  
`overfit_test()` bypassed `AugmentationManager` and passed raw hard integer labels directly to `SoftCrossEntropyLoss`.

**Fix:**
```python
# overfit_test() — convert hard labels to one-hot before loss
soft_labels = F.one_hot(labels, num_classes).float()
loss = criterion(logits, soft_labels)
```

**Learning:**  
When replacing `nn.CrossEntropyLoss` with a custom loss that changes the target format, audit every call site — not just the main loop. Test utilities are easy to miss.

---

## Summary Table

| # | Bug | Day | Severity | Root Cause |
|---|-----|-----|----------|------------|
| 1 | Grad-CAM blank after checkpoint | 6 | High | Hooks detach after `load_state_dict` |
| 2 | NaN loss at epoch 3 | 3 | Critical | AMP + LR too high → float16 overflow |
| 3 | `postgres://` Railway crash | 7B | High | Legacy URL scheme |
| 4 | `signal.alarm()` Windows crash | 7B | High | Unix-only API |
| 5 | DataLoader hang on Windows | 1 | Medium | `spawn` multiprocessing deadlock |
| 6 | Zero calories in nutrition CSV | 1 | Medium | USDA no match → fallback not triggered |
| 7 | Wrong class labels silently | 1 | Medium | `classes.txt` folder name mismatch |
| 8 | Soft label shape mismatch | 4 | Medium | Overfit test bypassed AugmentationManager |

---

*"My debugging process: hypothesis → minimal reproduction → fix → document.  
Every bug in this log was found through systematic investigation, not random trial-and-error."*
