"""
ECG Arrhythmia Classifier - Professional-grade heartbeat classification
using Machine Learning and MIT-BIH Arrhythmia Database.

Author: Rufus Pitta
License: MIT
"""

VERSION = "1.0.0"

import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pyttsx3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pickle
import wfdb
import time
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import random
 
BG     = '#0d1117'
PANEL  = '#161b22'
BORDER = '#30363d'
GREEN  = '#2ea043'
RED    = '#f85149'
ORANGE = '#d29922'
PURPLE = '#a371f7'
WHITE  = '#e6edf3'
GRAY   = '#8b949e'
 
LABEL_INFO = {
    'N': ('Normal Beat',         GREEN,  'NORMAL',    'Low'),
    'V': ('Ventricular Ectopic', RED,    'DANGEROUS', 'Critical'),
    'A': ('Atrial Premature',    ORANGE, 'WARNING',   'Moderate'),
    'L': ('Left Bundle Branch',  PURPLE, 'ABNORMAL',  'High'),
    'R': ('Right Bundle Branch', PURPLE, 'ABNORMAL',  'High'),
}

def _get_env_path(env_var: str, fallback_relative: str) -> str:
    """
    Get path from environment variable with fallback to relative path.
    
    Args:
        env_var: Environment variable name to check
        fallback_relative: Relative path fallback (e.g., './model/')
        
    Returns:
        str: Path from environment or fallback
        
    Raises:
        FileNotFoundError: If neither environment variable nor fallback exists
    """
    env_path = os.environ.get(env_var)
    if env_path and os.path.exists(env_path):
        return env_path
    
    if os.path.exists(fallback_relative):
        return fallback_relative
    
    raise FileNotFoundError(
        f"Path not found. Please set environment variable '{env_var}' to the correct directory. "
        f"Expected '{fallback_relative}' (relative) or set {env_var} in your environment."
    )

# Get paths with environment variable support
try:
    DATA_PATH = _get_env_path('MIT_BIH_PATH', './data/')
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    DATA_PATH = './data/'

try:
    MODEL_PATH = _get_env_path('ECG_MODEL_PATH', './model/')
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    MODEL_PATH = './model/'
 
def load_model() -> tuple:
    """
    Load pre-trained ECG classification model and dataset.
    
    Returns:
        tuple: (trained_model, features_array, labels_array)
        
    Raises:
        FileNotFoundError: If model or data files not found
        pickle.UnpicklingError: If files are corrupted
    """
    print("Loading real trained model...")
    model_file = os.path.join(MODEL_PATH, 'ecg_model.pkl')
    data_file = os.path.join(MODEL_PATH, 'ecg_data.pkl')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(data_file, 'rb') as f:
        X, y = pickle.load(f)
    print(f"Model loaded — {len(X)} real clinical beats available")
    return model, X, y
 
def extract_features(beats: np.ndarray) -> np.ndarray:
    """
    Extract 14 time-domain features from ECG beats.
    
    Args:
        beats: Array of ECG beat signals (180 samples each)
        
    Returns:
        np.ndarray: Extracted features (n_beats × 14 features)
    """
    beats = np.asarray(beats)
    if beats.ndim != 2 or beats.shape[1] != 180:
        raise ValueError(f'Expected beats with shape (n, 180), got {beats.shape}')
    if not np.isfinite(beats).all():
        raise ValueError('ECG beats must contain only finite numeric values')

    features = []
    for beat in beats:
        mean = np.mean(beat)
        std  = np.std(beat)
        mx   = np.max(beat)
        mn   = np.min(beat)
        features.append([
            mean, std, mx, mn,
            mx - mn,
            np.sum(beat**2),
            np.argmax(beat),
            np.argmax(beat) / 180.0,
            np.mean(((beat - mean) / (std + 1e-8))**3),
            np.mean(((beat - mean) / (std + 1e-8))**4),
            np.sum(np.diff(np.sign(beat)) != 0),
            np.sum(beat[:90]**2) / (np.sum(beat[90:]**2) + 1e-8),
            np.median(beat),
            np.percentile(beat, 75) - np.percentile(beat, 25),
        ])
    return np.array(features)
 
def load_beats_from_record(record_name: str) -> tuple:
    """
    Load ECG beats and labels from MIT-BIH database record.
    
    Args:
        record_name: MIT-BIH record identifier (e.g., '100')
        
    Returns:
        tuple: (beats_array, labels_array)
        
    Raises:
        FileNotFoundError: If record not found in DATA_PATH
    """
    record_path = os.path.join(DATA_PATH, record_name)
    if not os.path.exists(record_path + '.dat'):
        raise FileNotFoundError(f"Record not found: {record_path}")
    
    record     = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal     = record.p_signal[:, 0]
    window     = 90
    beats, labels = [], []
    keep = {'N', 'V', 'A', 'L', 'R'}
    for i, pos in enumerate(annotation.sample):
        if pos - window < 0 or pos + window > len(signal):
            continue
        if annotation.symbol[i] not in keep:
            continue
        beats.append(signal[pos - window: pos + window])
        labels.append(annotation.symbol[i])
    return np.array(beats), np.array(labels)
 
def get_diagnosis(label: str, beat_stats: dict) -> str:
    """
    Generate clinical diagnosis text based on beat classification.
    
    Args:
        label: Beat label ('N', 'V', 'A', or 'L')
        beat_stats: Dictionary with 'mean', 'peak' values
        
    Returns:
        str: Detailed clinical diagnosis text
    """
    name, _, status, risk = LABEL_INFO[label]
    diagnoses = {
        'N': f"""✓ NORMAL SINUS RHYTHM (NSR)
 
Classification : Normal beat detected
Mean           : {beat_stats['mean']:.4f} mV
Peak           : {beat_stats['peak']:.4f} mV
Status         : {status} | Risk: {risk}
 
CLINICAL ASSESSMENT:
Healthy heartbeat with normal cardiac conduction.
Regular rhythm and appropriate electrical characteristics.
 
RECOMMENDATION:
No intervention required. Continue routine monitoring.""",
 
        'V': f"""⚠ VENTRICULAR ECTOPIC BEAT (PVC)
 
Classification : DANGEROUS premature ventricular contraction
Mean           : {beat_stats['mean']:.4f} mV
Peak           : {beat_stats['peak']:.4f} mV (ELEVATED)
Status         : {status} | Risk: {risk}
 
CLINICAL ASSESSMENT:
Premature ventricular contraction originating from ventricles.
Shows abnormal conduction and high amplitude pattern.
 
RECOMMENDATION:
⚠ URGENT: Cardiologist evaluation required immediately.
Consider Holter monitoring and medication review.""",
 
        'A': f"""⚠ ATRIAL PREMATURE BEAT (PAC)
 
Classification : WARNING — early atrial beat
Mean           : {beat_stats['mean']:.4f} mV
Peak           : {beat_stats['peak']:.4f} mV
Status         : {status} | Risk: {risk}
 
CLINICAL ASSESSMENT:
Premature atrial contraction indicating atrial irritability.
Usually benign but may suggest underlying condition.
 
RECOMMENDATION:
Schedule cardiologist consultation.
Monitor for palpitations and lifestyle modifications.""",
 
        'L': f"""⚠ LEFT BUNDLE BRANCH BLOCK (LBBB)
 
Classification : ABNORMAL — conduction block detected
Mean           : {beat_stats['mean']:.4f} mV
Peak           : {beat_stats['peak']:.4f} mV (WIDENED)
Status         : {status} | Risk: {risk}
 
CLINICAL ASSESSMENT:
Impaired electrical conduction in the left ventricle.
Significant finding suggesting cardiac pathology.
 
RECOMMENDATION:
⚠ URGENT: Cardiologist consultation required immediately.
Full cardiac workup including echocardiogram needed.""",
    }
    return diagnoses.get(label, f"Classification: {name}\nStatus: {status}\nRisk: {risk}")

def generate_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Generate and save confusion matrix for model evaluation.
    
    Runs trained model on test set and creates confusion matrix visualization.
    Per-class and overall accuracy are printed to console.
    
    Args:
        model: Trained sklearn classifier
        X_test: Test feature set
        y_test: Test labels
        
    Returns:
        None (saves confusion_matrix.png)
    """
    print("Generating confusion matrix...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy metrics
    overall_acc = accuracy_score(y_test, y_pred)
    classes = ['N', 'V', 'A', 'L', 'R']
    
    print("\n" + "="*50)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*50)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
    
    # Per-class accuracy
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"{cls} ({LABEL_INFO[cls][0]:.<25}): {cls_acc:.4f} ({cls_acc*100:.2f}%)")
    print("="*50 + "\n")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Create visualization with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, ax=ax, 
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('ECG Arrhythmia Classification - Confusion Matrix\n', 
                 fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, facecolor='#0d1117')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
 
def generate_pdf_report(record: str, label: str, diagnosis: str, 
                       beat_stats: dict, inference_ms: float) -> str:
    """
    Generate PDF clinical report for classified beat.
    
    Args:
        record: MIT-BIH record ID
        label: Beat classification label
        diagnosis: Clinical diagnosis text
        beat_stats: Dictionary with 'mean', 'peak', 'energy'
        inference_ms: Inference time in milliseconds
        
    Returns:
        str: Filename of generated PDF
    """
    name, _, status, risk = LABEL_INFO[label]
    filename = f"ECG_Report_{record}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc    = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []
    story.append(Paragraph("ECG Arrhythmia Analysis Report", styles['Title']))
    story.append(Paragraph(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph("Built by Rufus Pitta | MIT-BIH Arrhythmia Database | Accuracy: 90%+", styles['Normal']))
    story.append(Spacer(1, 20))
    data = [
        ['Field', 'Value'],
        ['Record', record],
        ['Classification', f"{label} — {name}"],
        ['Status', status],
        ['Risk Level', risk],
        ['Mean Amplitude', f"{beat_stats['mean']:.4f} mV"],
        ['Peak Amplitude', f"{beat_stats['peak']:.4f} mV"],
        ['Signal Energy',  f"{beat_stats['energy']:.4f}"],
        ['Inference Time', f"{inference_ms:.2f} ms"],
    ]
    table = Table(data, colWidths=[200, 300])
    table.setStyle([
        ('BACKGROUND',     (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR',      (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID',           (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME',       (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ])
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("AI Clinical Diagnosis", styles['Heading2']))
    story.append(Paragraph(diagnosis.replace('\n', '<br/>'), styles['Normal']))
    doc.build(story)
    return filename
 
def speak(text: str) -> None:
    """
    Speak text using text-to-speech in background thread.
    
    Args:
        text: Text to speak
        
    Returns:
        None
    """
    def _speak() -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()
 
class ECGApp:
    """
    Tkinter GUI application for ECG arrhythmia classification.
    
    Displays ECG waveforms, runs ML model inference, generates reports,
    and provides voice alerts for dangerous arrhythmias.
    """
    
    def __init__(self, root: tk.Tk, model, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize ECG application GUI.
        
        Args:
            root: Tkinter root window
            model: Trained ML classifier
            X: Training features for evaluation
            y: Training labels for evaluation
        """
        self.root          = root
        self.model         = model
        self.X             = X
        self.y             = y
        self.current_beat  = None
        self.current_label = None
        self.diagnosis     = ""
        self.inference_ms  = 0.0
        self.record_cache  = {}
 
        root.title("ECG Arrhythmia Classifier — Rufus Pitta")
        root.configure(bg=BG)
        root.geometry("1400x900")
        self._build_ui()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        root.update()
        root.deiconify()
        root.lift()
 
    def _build_ui(self) -> None:
        """Build Tkinter user interface."""
        tf = tk.Frame(self.root, bg=BG)
        tf.pack(pady=10)
        tk.Label(tf, text="ECG ARRHYTHMIA CLASSIFIER",
                 font=('Consolas', 22, 'bold'), bg=BG, fg=GREEN).pack()
        tk.Label(tf,
                 text="Real Clinical Data  |  MIT-BIH Database  |  Built by Rufus Pitta  |  Accuracy: 90%+",
                 font=('Consolas', 10), bg=BG, fg=GRAY).pack()
 
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(pady=8)
        tk.Label(ctrl, text="Record:", bg=BG, fg=WHITE,
                 font=('Consolas', 11, 'bold')).pack(side='left', padx=5)
        self.record_var = tk.StringVar(value='100')
        ttk.Combobox(ctrl, textvariable=self.record_var, width=8,
                     values=['100','101','102','103','104','105',
                             '106','107','108','109','111','112']
                     ).pack(side='left', padx=5)
        tk.Label(ctrl, text="Beat #:", bg=BG, fg=WHITE,
                 font=('Consolas', 11, 'bold')).pack(side='left', padx=5)
        self.beat_var = tk.IntVar(value=0)
        tk.Spinbox(ctrl, from_=0, to=2000, textvariable=self.beat_var,
                   width=8, bg=PANEL, fg=WHITE).pack(side='left', padx=5)
 
        for text, cmd, color in [
            ("CLASSIFY",    self.classify,    GREEN),
            ("RANDOM",      self.random_beat, GRAY),
            ("VOICE",       self.voice_alert, ORANGE),
            ("PDF REPORT",  self.make_pdf,    PURPLE),
        ]:
            tk.Button(ctrl, text=text, command=cmd, bg=color, fg='black',
                      font=('Consolas', 10, 'bold'), relief='flat',
                      padx=12, pady=4).pack(side='left', padx=4)
 
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=10, pady=5)
 
        left = tk.Frame(main, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=2)
        left.pack(side='left', fill='both', expand=True, padx=(0, 8))
 
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(PANEL)
        self.ax.set_title('ECG Signal Waveform — Real MIT-BIH Data',
                          color=WHITE, fontsize=13, fontweight='bold')
        self.ax.tick_params(colors=GRAY)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
 
        right = tk.Frame(main, bg=BG, width=420)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
 
        res = tk.Frame(right, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1)
        res.pack(fill='x', pady=(0, 8))
        tk.Label(res, text="CLASSIFICATION RESULT",
                 font=('Consolas', 10, 'bold'), bg=PANEL, fg=GRAY).pack(pady=(6, 2))
        self.result_label = tk.Label(res, text="—",
                                     font=('Consolas', 36, 'bold'), bg=PANEL, fg=WHITE)
        self.result_label.pack()
        self.result_name = tk.Label(res, text="Select record and click CLASSIFY",
                                    font=('Consolas', 12), bg=PANEL, fg=GRAY,
                                    wraplength=400, justify='center')
        self.result_name.pack(pady=4)
        self.status_label = tk.Label(res, text="",
                                     font=('Consolas', 11, 'bold'), bg=PANEL, fg=WHITE)
        self.status_label.pack(pady=(0, 8))
 
        stats = tk.Frame(right, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        stats.pack(fill='x', pady=(0, 8))
        tk.Label(stats, text="SIGNAL STATISTICS",
                 font=('Consolas', 10, 'bold'), bg=PANEL, fg=GRAY).pack(pady=(6, 2))
        self.stats_text = tk.Label(stats, text="—",
                                   font=('Consolas', 10), bg=PANEL,
                                   fg=WHITE, justify='left')
        self.stats_text.pack(padx=8, pady=(0, 8))
 
        tk.Label(right, text="AI DIAGNOSIS",
                 font=('Consolas', 10, 'bold'), bg=BG, fg=GRAY).pack(anchor='w', pady=(8, 0))
        self.diag_box = scrolledtext.ScrolledText(
            right, height=14, width=52,
            font=('Consolas', 9), bg=PANEL, fg=WHITE,
            insertbackground=WHITE, relief='flat', wrap='word')
        self.diag_box.pack(fill='both', expand=True, pady=(2, 0))
 
        self.status_bar = tk.Label(
            self.root, text="Ready — Select a record and click CLASSIFY",
            font=('Consolas', 9), bg=PANEL, fg=GRAY, anchor='w')
        self.status_bar.pack(fill='x', padx=10, pady=(4, 8))
 
    def _get_beats(self, record_name: str) -> tuple:
        """
        Get beats from record, with caching.
        
        Args:
            record_name: MIT-BIH record ID
            
        Returns:
            tuple: (beats_array, labels_array) or (None, None) on error
        """
        if record_name not in self.record_cache:
            self.status_bar.config(text=f"Loading record {record_name}...")
            self.root.update()
            try:
                beats, labels = load_beats_from_record(record_name)
                self.record_cache[record_name] = (beats, labels)
            except Exception as e:
                self.status_bar.config(text=f"Error: {e}")
                return None, None
        return self.record_cache[record_name]
 
    def classify(self) -> None:
        """Classify selected beat and display results."""
        rec      = self.record_var.get()
        beat_idx = self.beat_var.get()
        beats, labels = self._get_beats(rec)
        if beats is None:
            return
 
        beat_idx = beat_idx % len(beats)
        beat     = beats[beat_idx]
        actual   = labels[beat_idx]
 
        feat  = extract_features([beat])
        start = time.time()
        pred  = self.model.predict(feat)[0]
        ms    = (time.time() - start) * 1000
        self.inference_ms = ms
 
        name, color, status, risk = LABEL_INFO[pred]
        self.current_beat  = beat
        self.current_label = pred
 
        beat_stats = {
            'mean':   float(np.mean(beat)),
            'peak':   float(np.max(beat)),
            'energy': float(np.sum(beat**2)),
        }
 
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        self.ax.plot(beat, color=color, linewidth=2.5)
        self.ax.axvline(x=90, color=WHITE, linestyle='--', alpha=0.4)
        self.ax.grid(True, alpha=0.2, color=GRAY)
        self.ax.set_title(
            f'{name}  |  {status}  |  Inference: {ms:.1f}ms',
            color=color, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Samples', color=GRAY, fontsize=9)
        self.ax.set_ylabel('Amplitude (mV)', color=GRAY, fontsize=9)
        self.ax.tick_params(colors=GRAY, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.fig.patch.set_facecolor(BG)
        self.canvas.draw()
 
        self.result_label.config(text=pred, fg=color)
        self.result_name.config(text=name, fg=color)
        self.status_label.config(text=f"{status}  |  Risk: {risk}", fg=color)
        self.stats_text.config(text=(
            f"Mean        : {beat_stats['mean']:.5f} mV\n"
            f"Peak        : {beat_stats['peak']:.5f} mV\n"
            f"Energy      : {beat_stats['energy']:.5f}\n"
            f"Actual      : {actual}\n"
            f"Inference   : {ms:.2f} ms"
        ))
 
        self.diagnosis = get_diagnosis(pred, beat_stats)
        self.diag_box.config(state='normal')
        self.diag_box.delete('1.0', tk.END)
        self.diag_box.insert(tk.END, self.diagnosis)
        self.diag_box.config(state='disabled')
 
        if pred == 'V':
            speak(f"Warning. Dangerous arrhythmia detected. {name}. Immediate attention required.")
 
        self.status_bar.config(
            text=f"Record {rec} | Beat {beat_idx} | Predicted: {pred} | Actual: {actual} | {ms:.2f}ms")
 
    def random_beat(self) -> None:
        """Select and classify a random beat."""
        self.record_var.set(random.choice(
            ['100','101','102','103','104','105',
             '106','107','108','109','111','112']))
        self.beat_var.set(random.randint(0, 500))
        self.classify()
 
    def voice_alert(self) -> None:
        """Speak current classification result."""
        if not self.current_label:
            speak("No beat classified yet.")
            return
        name, _, status, risk = LABEL_INFO[self.current_label]
        speak(f"ECG result. {name}. Status {status}. Risk {risk}.")
 
    def make_pdf(self) -> None:
        """Generate PDF report for current classification."""
        if self.current_beat is None:
            self.status_bar.config(text="Classify a beat first.")
            return
        self.status_bar.config(text="Generating PDF...")
        self.root.update()
        beat_stats = {
            'mean':   float(np.mean(self.current_beat)),
            'peak':   float(np.max(self.current_beat)),
            'energy': float(np.sum(self.current_beat**2)),
        }
        filename = generate_pdf_report(
            self.record_var.get(),
            self.current_label,
            self.diagnosis,
            beat_stats,
            self.inference_ms
        )
        self.status_bar.config(text=f"PDF saved: {filename}")
        speak("PDF report generated successfully.")
 
if __name__ == '__main__':
    print(f"ECG Arrhythmia Classifier v{VERSION}")
    print("Starting application...")
    try:
        print("Loading model...")
        model, X, y = load_model()
        print("Model loaded successfully!")
        print("Creating Tkinter window...")
        root = tk.Tk()
        print("Tkinter window created successfully")
        print("Initializing ECG app...")
        app  = ECGApp(root, model, X, y)
        print("App initialized successfully!")
        print("Starting main event loop...")
        root.mainloop()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
