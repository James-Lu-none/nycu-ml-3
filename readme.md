# NYCU-IAlI-ML2025 - Recurrent Neural Networks (Taiwanese Speech Recognition)

## results

### openai_whisper_small

[eval with WER](history/openai_whisper_small_2025-11-07T15-06-55_2025-11-07T15-58-32.csv)
submit score: 9.39393

[eval with WER & apply lexicon](history/openai_whisper_small_2025-11-07T15-06-55_2025-11-07T19-33-53.csv)
submit score: 9.87878

eval with normalized Levenshtein distance

### openai_whisper_medium


### whisper note

in a model dir:
.
├── added_tokens.json
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── normalizer.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── training_args.bin
└── vocab.json

#### terms
- decoder_start_token_id
in original tokenizer, model.config.decoder_start_token_id is "<|startoftranscript|>": 50258, in my custom tokenizer, i use `<s>` and `</s>` as bos and eos token

- vocab
- beam_num
num_beams means how many candidates are considered in each next possible tokens
ex: Candidate next tokens:
"ba" (score 0.6)
"pa" (score 0.3)
"ka" (score 0.1)
Greedy (num_beams=1): choose "ba"
Greedy (num_beams=2): choose ["ba","pa"]
- tokenizer types (in tokenizer_config.json)
word level (each valid word is a token)
sub-word level (Byte pair encoding, word piece, sentence piece)
character level (each char is a token)



# Data Collator for Speech-to-Text: Complete Explanation

## Overview

A **Data Collator** combines individual samples into uniform batches for training.

---

## The Problem

### Individual Samples (Variable Sizes)

```
Sample 1:
  input_features: [80, 523]   ← 523 time steps (0.52 sec audio)
  labels: [15]                ← 15 tokens

Sample 2:
  input_features: [80, 1842]  ← 1842 time steps (1.84 sec audio)
  labels: [28]                ← 28 tokens

Sample 3:
  input_features: [80, 891]   ← 891 time steps
  labels: [22]                ← 22 tokens
```

**Problem**: Can't stack these into a batch tensor because shapes don't match!

---

## The Solution: Padding

### Step 1: Pad Audio Features

Pad all audio to **3000 time steps** (max length):

```
Sample 1: [80, 523]  → [80, 3000]  (add 2477 zeros)
Sample 2: [80, 1842] → [80, 3000]  (add 1158 zeros)
Sample 3: [80, 891]  → [80, 3000]  (add 2109 zeros)

Result: Stack into batch [3, 80, 3000]
```

### Step 2: Pad Text Labels

Pad all labels to **max length in batch** (here: 28):

```
Sample 1: [15] → [15, -100, -100, ..., -100]  (add 13 × -100)
Sample 2: [28] → [28]                          (already longest)
Sample 3: [22] → [22, -100, -100, ..., -100]  (add 6 × -100)

Result: Stack into batch [3, 28]
```

**Why -100?** PyTorch's loss function **ignores** positions with -100, so padding doesn't affect training!

---

## Code Walkthrough

```python
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any                    # Your WhisperProcessor
    decoder_start_token_id: int      # BOS token ID (2 for you)

    def __call__(self, features):
        # features = list of samples from your dataset
        # Example: [sample_1, sample_2, sample_3]
```

### Part 1: Extract Data

```python
        input_features = [feature["input_features"] for feature in features]
        # → [tensor[80,523], tensor[80,1842], tensor[80,891]]
        
        label_features = [feature["labels"] for feature in features]
        # → [tensor[15], tensor[28], tensor[22]]
        
        batch_size = len(input_features)  # → 3
```

### Part 2: Pad Audio Features

```python
        feature_dim = 80          # Mel spectrogram has 80 frequency bins
        max_length = 3000         # Maximum time steps (30 seconds @ 16kHz)
        
        # Create empty tensor filled with zeros
        input_features_padded = torch.zeros(batch_size, feature_dim, max_length)
        # Shape: [3, 80, 3000]
        
        for i, features_tensor in enumerate(input_features):
            # features_tensor shape: [80, 523] for sample 1
            actual_feature_dim = features_tensor.shape[0]    # 80
            actual_time_steps = features_tensor.shape[1]     # 523
            
            copy_feature_dim = min(actual_feature_dim, feature_dim)   # 80
            copy_time_steps = min(actual_time_steps, max_length)      # 523
            
            # Copy actual data into the padded tensor
            input_features_padded[i, :copy_feature_dim, :copy_time_steps] = \
                features_tensor[:copy_feature_dim, :copy_time_steps]
            
            # Result: [80, 523] copied, rest stays zero (padding)
```

**Visual Example:**

```
Original:  [80, 523]
Padded:    [80, 3000]
           ├─────┬──────┐
           │ 523 │ 2477 │
           │ real│  0s  │
           └─────┴──────┘
```

### Part 3: Pad Text Labels

```python
        max_label_length = max(len(label) for label in label_features)
        # → 28 (longest label in this batch)
        
        labels_padded = []
        for label in label_features:
            # label is tensor[15] for sample 1
            padding_length = max_label_length - len(label)  # 28 - 15 = 13
            
            padded_label = torch.cat([
                label,  # Original: [2, 45, 67, ..., 3]  (15 tokens)
                torch.full((padding_length,), -100, dtype=label.dtype)  # [13 × -100]
            ])
            # Result: [2, 45, 67, ..., 3, -100, -100, ..., -100]  (28 tokens)
            
            labels_padded.append(padded_label)
        
        labels = torch.stack(labels_padded)  # Shape: [3, 28]
```

### Part 4: Remove BOS Token from Labels (if present)

```python
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
```

**Why?** During training, the model automatically adds BOS token. If your labels already have it, you'd have it twice:

```
Labels with BOS:    [<s>, tok1, tok2, ..., </s>]
Model adds BOS:     [<s>, <s>, tok1, tok2, ..., </s>]  ← Wrong!

So remove it:       [tok1, tok2, ..., </s>]
Model adds BOS:     [<s>, tok1, tok2, ..., </s>]       ← Correct!
```

### Part 5: Handle Pad Tokens

```python
        pad_token_id = 0  # Your <pad> token ID
        labels[labels == pad_token_id] = -100
```

**Why?** If your tokenizer added `<pad>` (ID=0) instead of -100, replace it so the loss ignores it.

### Part 6: Return Batch

```python
        batch = {
            "input_features": input_features_padded,  # [3, 80, 3000]
            "labels": labels                           # [3, 28]
        }
        return batch
```

---

## Key Concepts

### 1. **Padding**

Adding zeros (for audio) or -100 (for text) to make all samples the same size.

### 2. **-100 in Labels**

Special value that tells PyTorch: "Don't calculate loss for this position."

### 3. **Batch**

Multiple samples processed together for efficient GPU computation.

### 4. **Mel Spectrogram Features**

Audio converted to frequency representation:

- **80 frequency bins** (vertical axis)
- **Variable time steps** (horizontal axis)
- Padded to **3000 time steps** max

---

## Example Flow

```
Dataset returns 3 samples:
├─ Sample 1: audio[80,523],  labels[15]
├─ Sample 2: audio[80,1842], labels[28]
└─ Sample 3: audio[80,891],  labels[22]

↓ DataCollator processes ↓

Batch ready for training:
├─ input_features: [3, 80, 3000]  (all padded to 3000)
└─ labels: [3, 28]                 (all padded to 28 with -100)

↓ Model trains ↓

Loss calculated only on real tokens (ignoring -100 positions)
```

---

## Related Terms

| Term | Meaning |
|------|---------|
| **Batch** | Multiple samples processed together |
| **Padding** | Adding filler values to match sizes |
| **Collate** | Combining items into a batch |
| **Feature Dim** | Number of mel frequency bins (80) |
| **Time Steps** | Audio duration in frames (~100 per second) |
| **BOS** | Beginning of Sequence token (`<s>`) |
| **EOS** | End of Sequence token (`</s>`) |
| **Pad Token** | Special token for padding (`<pad>`) |
| **-100** | Special label value to ignore in loss |

---

## Why You Need This

Without a data collator:

```python
# ❌ This fails:
batch = torch.stack([sample1["input_features"], sample2["input_features"]])
# RuntimeError: stack expects each tensor to be equal size
```

With a data collator:

```python
# ✅ This works:
batch = data_collator([sample1, sample2, sample3])
# Returns properly padded tensors ready for training
```
