import argparse
import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import numpy as np
import re
import functools
from tqdm import tqdm

TRAIN_AUDIO = "data/train/noisy-train"
TRAIN_CSV = "data/train/train-toneless.csv"
OUTPUT_ROOT = "visualization"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

class visualization:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.processor = None
        self.model = None
        self.valid_words = None

    def load_model(self, model_dir):
        print(f"Loading model from: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe_audio(self, audio_path, max_length_sec=30):
        audio, sr = librosa.load(audio_path, sr=16000)

        max_samples = max_length_sec * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.model.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_length=225,
                num_beams=3,
            )

        sentence = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        sentence = self.correct_taiwanese_sentence(sentence)

        return sentence

    def transcribe_directory(self):
        rows = []

        wav_files = sorted([f for f in os.listdir(TRAIN_AUDIO) if f.lower().endswith(".wav")])
        for file in tqdm(wav_files, desc="Processing audio", unit="file"):
            full_path = os.path.join(TRAIN_AUDIO, file)
            audio_id = os.path.splitext(file)[0]

            sentence = self.transcribe_audio(full_path)
            rows.append({"id": audio_id, "sentence": sentence})

        df = pd.DataFrame(rows)
        return df
    import argparse
import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import numpy as np
import re
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

TRAIN_AUDIO = "data/train/noisy-train"
TRAIN_CSV = "data/train/train-toneless.csv"
OUTPUT_ROOT = "visualization"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

class visualization:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.processor = None
        self.model = None
        self.valid_words = None

    def load_model(self, model_dir):
        print(f"Loading model from: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe_audio(self, audio_path, max_length_sec=30):
        audio, sr = librosa.load(audio_path, sr=16000)

        max_samples = max_length_sec * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.model.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_length=225,
                num_beams=3,
            )

        sentence = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return sentence

    def transcribe_directory(self):
        rows = []

        wav_files = sorted([f for f in os.listdir(TRAIN_AUDIO) if f.lower().endswith(".wav")])
        for file in tqdm(wav_files, desc="Processing audio", unit="file"):
            full_path = os.path.join(TRAIN_AUDIO, file)
            audio_id = os.path.splitext(file)[0]

            sentence = self.transcribe_audio(full_path)
            rows.append({"id": audio_id, "sentence": sentence})

        df = pd.DataFrame(rows)
        return df
    
    def generate_confusion_matrix(self, predicted_df, output_path):
        """Generate character-level confusion matrix visualization"""
        # Load ground truth
        gt_df = pd.read_csv(TRAIN_CSV)
        
        # Merge predicted and ground truth on id
        merged = predicted_df.merge(gt_df, on='id', suffixes=('_pred', '_gt'))
        
        # Collect all character pairs
        true_chars = []
        pred_chars = []
        
        for _, row in merged.iterrows():
            gt_sentence = row['sentence_gt']
            pred_sentence = row['sentence_pred']
            
            # Align characters (simple approach: pad to same length)
            max_len = max(len(gt_sentence), len(pred_sentence))
            gt_padded = gt_sentence.ljust(max_len)
            pred_padded = pred_sentence.ljust(max_len)
            
            for gt_char, pred_char in zip(gt_padded, pred_padded):
                if gt_char != ' ' and pred_char != ' ':
                    true_chars.append(gt_char)
                    pred_chars.append(pred_char)
        
        # Get most common characters for visualization
        all_chars = list(set(true_chars + pred_chars))
        char_counts = Counter(true_chars)
        top_chars = [char for char, _ in char_counts.most_common(30)]
        
        # Filter to top characters
        filtered_true = []
        filtered_pred = []
        for t, p in zip(true_chars, pred_chars):
            if t in top_chars and p in top_chars:
                filtered_true.append(t)
                filtered_pred.append(p)
        
        # Generate confusion matrix
        cm = confusion_matrix(filtered_true, filtered_pred, labels=top_chars)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create visualization
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm_normalized, 
                    annot=False,
                    fmt='.2f',
                    cmap='YlOrRd',
                    xticklabels=top_chars,
                    yticklabels=top_chars,
                    cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title('Character-Level Confusion Matrix (Top 30 Characters)', fontsize=16, pad=20)
        plt.xlabel('Predicted Character', fontsize=12)
        plt.ylabel('True Character', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
        plt.close()
        
        # Also save a summary CSV
        summary_path = output_path.replace('.png', '_summary.csv')
        summary_data = []
        for i, true_char in enumerate(top_chars):
            for j, pred_char in enumerate(top_chars):
                if cm[i, j] > 0:
                    summary_data.append({
                        'true_char': true_char,
                        'pred_char': pred_char,
                        'count': cm[i, j],
                        'normalized': cm_normalized[i, j]
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('count', ascending=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"Confusion summary saved to: {summary_path}")
    
    def run(self):
        self.load_model(self.model_dir)
        model_choice = f"{self.model_dir.split('/')[-2]}_{self.model_dir.split('/')[-1]}"
        df = self.transcribe_directory()
        output_confusion = os.path.join(OUTPUT_ROOT, f"{model_choice}.png")
        self.generate_confusion_matrix(df, output_confusion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved.")
    args = parser.parse_args()
    visualizer = visualization(args.model_dir)
    visualizer.run()