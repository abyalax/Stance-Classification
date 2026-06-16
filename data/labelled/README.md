# 📋 Labeling Pipeline Status

## 🎯 Current Stage: Manual Annotation

### ✅ Completed
- **Data Collection**: 5.186 comments (3.903 TikTok + 1.283 YouTube)
- **Preprocessing**: 3.542 comments (cleaned & filtered)
- **Labeling Preparation**: 500 samples ready for annotation
- **Label Studio Setup**: Running on `http://localhost:8080`

### 🔄 In Progress
- **Manual Labeling**: 500 samples pending annotation
- **Quality Control**: Pending multi-annotator review

### ⏳ Next Steps
1. **Complete Manual Labeling**: All 500 samples
2. **Quality Validation**: Cohen's Kappa calculation
3. **Export Labeled Data**: Convert to training format
4. **Fine-Tuning BERT**: Train stance classifier

---

## 📁 Available Files

### 📥 Labeling Resources
- `labeling_dataset.json` - 500 samples for Label Studio import
- `labeling_dataset.csv` - Same data in CSV format
- `annotation_guidelines.md` - MBG stance classification rules
- `labeling_analysis.json` - Dataset statistics
- `LABELING_GUIDE.md` - Complete labeling workflow

### 📊 Dataset Statistics
- **Total Samples**: 500
- **Platform Distribution**: 74% TikTok, 26% YouTube
- **Text Length**: Mean 9.8 words, Median 7 words
- **Quality Filters**: 3-50 words, non-spam only

---

## 🎯 Quick Start

### 1. Access Label Studio
```
URL: http://localhost:8080
Status: Running ✅
```

### 2. Import Dataset
1. Go to Project → Import
2. Upload: `labeling_dataset.json`
3. Configure: Text field = `final_text`

### 3. Start Labeling
1. Follow: `annotation_guidelines.md`
2. Apply: 8-label schema
3. Track: Progress and quality

### 4. Quality Control
1. Multiple annotators: 2-3 people
2. Overlap samples: 50-100 for agreement
3. Target Cohen's Kappa: κ ≥ 0.6

---

## 📖 Documentation

### 📋 View Complete Guide
- **Full Workflow**: `LABELING_GUIDE.md`
- **Annotation Rules**: `annotation_guidelines.md`
- **Research Context**: `../research_context.md`

### 🔗 Pipeline Integration
- **Previous Stage**: Preprocessing (`../preprocessed/`)
- **Next Stage**: Fine-tuning (`../pipelines/04_finetuning_bert.py`)

---

## 🚨 Important Notes

### 📝 Labeling Quality
- **Consistency**: Apply same criteria across all samples
- **Context**: Focus on MBG program stance
- **Platform**: Consider TikTok vs YouTube differences
- **Documentation**: Add notes for ambiguous cases

### 📊 Success Metrics
- **Completion Rate**: 100% of 500 samples labeled
- **Agreement Score**: Cohen's Kappa ≥ 0.6
- **Label Distribution**: Balanced across 8 categories
- **Average Confidence**: ≥ 3.0/5.0

---

## 📞 Help & Resources

### 📖 Reference Materials
- [Labeling Guide](LABELING_GUIDE.md) - Complete workflow
- [Annotation Guidelines](labeling_package/annotation_guidelines.md) - Detailed rules
- [Research Context](../research_context.md) - Project background

### 🔧 Technical Support
- **Docker Status**: `docker ps` to check container
- **Log Issues**: `docker logs label-studio` for troubleshooting
- **Port Access**: Ensure 8080 is available

---

*Pipeline Stage: Labeling Preparation → Manual Annotation*
*Progress: 500/500 samples ready for labeling*
*Next: Complete manual annotation → Quality validation*
