# 🏷️ Labeling Guide: From Label Studio to Fine-Tuning BERT

## 📋 Table of Contents

1. [🚀 Setup Label Studio](#setup-label-studio)
2. [📥 Import Dataset](#import-dataset)
3. [🎯 Configure Labeling Interface](#configure-labeling-interface)
4. [🏷️ Manual Labeling Process](#manual-labeling-process)
5. [📊 Quality Control](#quality-control)
6. [📤 Export Labeled Data](#export-labeled-data)
7. [🔧 Prepare for Fine-Tuning](#prepare-for-fine-tuning)
8. [🤖 Next: Fine-Tuning BERT](#next-fine-tuning-bert)

---

## 🚀 Setup Label Studio

### Access Label Studio

1. **URL**: `http://localhost:8080`
2. **Sign up** (fresh install) atau **Login** (existing account)

### First Time Setup

1. **Create Account**: Email + Password
2. **Create Organization**: Personal/Organization name
3. **Create Project**:
   - **Name**: "MBG Stance Classification"
   - **Type**: "Text Classification"

---

## 📥 Import Dataset

### Import JSON File

1. **Go to**: Project → Import
2. **Select**: "JSON" format
3. **Upload**: `data/labelled/labeling_dataset.json`
4. **Configure**:
   - **Text Field**: `final_text` (primary text for labeling)
   - **Meta Fields**: `source`, `created_at`, `likes_count`, `reply_count`

### Import Settings

```json
{
  "text": "final_text",
  "meta": {
    "source": "source",
    "created_at": "created_at",
    "likes_count": "likes_count",
    "reply_count": "reply_count"
  }
}
```

---

## 🎯 Configure Labeling Interface

### Label Schema Setup

1. **Go to**: Settings → Labeling Interface
2. **Add HTML Template**: Use custom template
3. **Configure Labels**:

#### Label Options

```xml
<Choices>
  <Choice name="FAVOR" value="FAVOR">FAVOR - Mendukung program MBG</Choice>
  <Choice name="AGAINST" value="AGAINST">AGAINST - Menentang program MBG</Choice>
  <Choice name="PRO_GOV" value="PRO_GOV">PRO_GOV - Mendukung pemerintah</Choice>
  <Choice name="CONTRA_GOV" value="CONTRA_GOV">CONTRA_GOV - Menentang pemerintah</Choice>
  <Choice name="CONDITIONAL" value="CONDITIONAL">CONDITIONAL - Dukung bersyarat</Choice>
  <Choice name="SUGGESTION" value="SUGGESTION">SUGGESTION - Saran konstruktif</Choice>
  <Choice name="DISCUSSION" value="DISCUSSION">DISCUSSION - Diskusi antar pengguna</Choice>
  <Choice name="OFF_TOPIC" value="OFF_TOPIC">OFF_TOPIC - Tidak relevan</Choice>
</Choices>
```

#### Rating Scale

```xml
<Rating name="confidence" title="Confidence Level">
  1 - Very Low
  2 - Low
  3 - Medium
  4 - High
  5 - Very High
</Rating>
```

#### Notes Field

```xml
<Textarea name="notes" title="Notes (Optional)" rows="3"/>
```

---

## 🏷️ Manual Labeling Process

### Annotation Guidelines

#### 📖 Reference Documentation

1. **Open**: `../research_context.md` (Section 9. Annotation Guidelines)
2. **Follow**: MBG Stance Classification rules
3. **Apply**: Decision hierarchy consistently

#### 🎯 Labeling Workflow

1. **Read Original Text**: Context understanding
2. **Check Cleaned Text**: For clarity
3. **Review Final Text**: For stance analysis
4. **Apply Decision Tree**:
   ```
   OFF_TOPIC → DISCUSSION → PRO_GOV/CONTRA_GOV → FAVOR/AGAINST → CONDITIONAL → SUGGESTION
   ```

#### 📝 Labeling Tips

- **Consistency**: Apply same criteria across all samples
- **Context**: Consider MBG program and Prabowo statements
- **Platform**: Account for TikTok vs YouTube language styles
- **Notes**: Add comments for ambiguous cases
- **Confidence**: Rate annotation certainty (1-5)

---

## 📊 Quality Control

### Multi-Annotator Setup

1. **Add Annotators**: Invite 2-3 team members
2. **Assign Samples**: Distribute 500 samples across annotators
3. **Track Progress**: Monitor completion rates

### Inter-Annotator Agreement

1. **Overlap Samples**: 50-100 samples labeled by multiple annotators
2. **Calculate Cohen's Kappa**:
   - Target: κ ≥ 0.6 (substantial agreement)
   - Tool: Python `sklearn.metrics.cohen_kappa_score`
3. **Resolve Disagreements**: Discuss conflicting labels

### Quality Metrics

- **Agreement Rate**: % of samples with same labels
- **Kappa Score**: Inter-annotator reliability
- **Average Confidence**: Annotator certainty
- **Label Distribution**: Balance across 8 categories

---

## 📤 Export Labeled Data

### Export Options

1. **Go to**: Export → Choose format
2. **Recommended**: JSON (for processing)
3. **Include**: Annotations + metadata

### Export Format

```json
{
  "annotations": [...],
  "predictions": [...],
  "task_data": [...]
}
```

### Save Location

- **Output**: `data/labelled/labeled_dataset.json`
- **Backup**: Keep original `labeling_dataset.json`

---

## 🔧 Prepare for Fine-Tuning

### Convert to Training Format

1. **Parse JSON**: Extract annotations and text
2. **Create DataFrame**:

   ```python
   import pandas as pd

   # Load labeled data
   df = pd.read_json('data/labelled/labeled_dataset.json')

   # Extract labels and text
   training_data = []
   for item in df:
       if item.get('annotations'):
           label = item['annotations'][0]['result'][0]['value']
           text = item['data']['final_text']
           training_data.append({'text': text, 'label': label})

   df_train = pd.DataFrame(training_data)
   ```

3. **Save**: `data/labelled/labeled_dataset.csv`

### Label Mapping

```python
label_map = {
    'FAVOR': 0,
    'AGAINST': 1,
    'PRO_GOV': 2,
    'CONTRA_GOV': 3,
    'CONDITIONAL': 4,
    'SUGGESTION': 5,
    'DISCUSSION': 6,
    'OFF_TOPIC': 7
}
```

### Data Split

```python
from sklearn.model_selection import train_test_split

# Split dataset
train_df, temp_df = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

## 🤖 Next: Fine-Tuning BERT

### Prerequisites

- ✅ **Labeled Dataset**: `data/labelled/labeled_dataset.csv`
- ✅ **Quality Metrics**: Cohen's Kappa ≥ 0.6
- ✅ **Data Balance**: Check label distribution

### Run Fine-Tuning Pipeline

```bash
# Execute fine-tuning
make finetune-force
```

### Expected Output

- **Model**: `models/indobert_mbg_classifier.pt`
- **Metrics**: F1-score, accuracy, confusion matrix
- **Ready**: For inference on new comments

---

## 📁 File Structure After Labeling

```
data/
├── labelled/
│   ├── labeling_dataset.json          # Original (unlabeled)
│   ├── labeled_dataset.json          # Export from Label Studio
│   ├── labeled_dataset.csv           # Training format
│   ├── labeling_package/             # Labeling prep files
│   │   ├── annotation_guidelines.md
│   │   └── labeling_analysis.json
│   └── LABELING_GUIDE.md         # This file
├── preprocessed/
│   └── comments_processed_2.csv
└── scrapped/
    ├── tiktok_comments_raw.csv
    └── youtube_comments_raw.csv
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Import Fails**: Check JSON format
2. **Label Interface**: Verify HTML template syntax
3. **Export Empty**: Ensure annotations are saved
4. **Memory Issues**: Process data in batches

### Solutions

- **Validate JSON**: Use JSON linting tools
- **Test Template**: Preview before applying
- **Backup Data**: Keep original files
- **Monitor Resources**: Check Docker memory

---

## 📞 Support

### Documentation

- **Annotation Guidelines**: `data/labelled/labeling_package/annotation_guidelines.md`
- **Research Context**: `research_context.md`
- **Pipeline Status**: `data/pipeline_status.json`

### Next Steps

1. **Complete Labeling**: 500 samples with quality control
2. **Export Data**: Convert to training format
3. **Run Fine-Tuning**: Execute BERT training pipeline
4. **Evaluate Model**: Test on held-out data

---

_Last Updated: 2026-03-25_
_Pipeline Stage: Labeling Preparation → Manual Annotation → Fine-Tuning_
