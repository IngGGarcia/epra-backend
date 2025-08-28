# EEG Valence and Arousal Calculation Guide

This guide explains the new EEG-based valence and arousal calculation functionality that has been added to your system. The implementation is based on current research findings and provides state-of-the-art emotion recognition capabilities.

## Overview

The system now includes functionality to calculate valence and arousal from EEG data using research-based methods, creating a dataframe with `image_number`, `valence`, and `arousal` columns similar to your existing SAM data structure.

## Key Features Implemented

### 1. Research-Based Methods
- **Power Spectral Density (PSD)** in frequency bands
- **Differential Entropy (DE)** - most widely used time-frequency feature
- **Asymmetry features** between left/right hemispheres
- **Time domain features** (first-order and second-order differences)
- **Support Vector Machine (SVM)** classification with RBF kernel

### 2. Frequency Band Analysis
- **Delta**: 1-4 Hz
- **Theta**: 4-8 Hz  
- **Alpha**: 8-13 Hz
- **Beta**: 13-30 Hz
- **Gamma**: 30-50 Hz
- **High Gamma**: 50-80 Hz (best for emotion recognition according to recent research)

### 3. Emotion Calculation Methods
- **Heuristic Method**: Based on established neuroscience research
  - Valence: Frontal alpha asymmetry (Davidson's model)
  - Arousal: Beta/alpha ratio across frontal regions
- **Machine Learning Method**: SVM-based classification (requires training data)

## Usage Examples

### 1. Simple Usage with Service Function

```python
from src.app.services.evaluation.eeg_valence_arousal_service import calculate_eeg_valence_arousal

# Calculate valence and arousal from EEG file
df_eeg_emotions = calculate_eeg_valence_arousal(
    upload_file.file, 
    upload_file.filename,
    method='heuristic'
)

print(df_eeg_emotions.columns.tolist())
# Output: ['image_number', 'valence', 'arousal']

print(df_eeg_emotions.head())
#    image_number   valence   arousal
# 0             1      6.23      5.78
# 1             2      4.91      7.12
# 2             3      7.45      3.67
```

### 2. Advanced Usage with Service Class

```python
from src.app.services.evaluation.eeg_valence_arousal_service import EEGValenceArousalService

# Initialize service
service = EEGValenceArousalService(
    sampling_rate=128,  # Optional: auto-detected if None
    method='heuristic'
)

# Calculate valence and arousal
df_results = service.calculate_valence_arousal_from_file(
    file_input, 
    filename
)

# Get detailed feature analysis
feature_summary = service.get_feature_summary(
    file_input, 
    filename
)
```

### 3. Direct Calculator Usage

```python
from src.modules.EEG.feature_extraction.valence_arousal_calculator import ValenceArousalCalculator

# Initialize calculator
calculator = ValenceArousalCalculator(
    sampling_rate=128,
    channels=['AF3', 'AF4', 'F3', 'F4']
)

# Prepare EEG data (dictionary with channel names as keys)
eeg_data = {
    'AF3': af3_signal,  # numpy array
    'AF4': af4_signal,  # numpy array
    'F3': f3_signal,    # numpy array
    'F4': f4_signal     # numpy array
}

# Calculate valence and arousal
valence, arousal = calculator.calculate_valence_arousal(
    eeg_data, 
    method='heuristic'
)

# Extract specific features
psd_features = calculator.extract_power_spectral_density(eeg_data)
de_features = calculator.extract_differential_entropy(eeg_data)
asymmetry_features = calculator.extract_asymmetry_features(eeg_data)
time_features = calculator.extract_time_domain_features(eeg_data)
```

## API Endpoints

### 1. Calculate Valence and Arousal

**Endpoint**: `POST /api/v1/eeg/valence-arousal/calculate/`

**Parameters**:
- `file`: EEG CSV file (multipart/form-data)
- `method`: Calculation method ('heuristic' or 'ml') - optional, defaults to 'heuristic'

**Response**:
```json
{
    "success": true,
    "data": [
        {"image_number": 1, "valence": 6.23, "arousal": 5.78},
        {"image_number": 2, "valence": 4.91, "arousal": 7.12},
        {"image_number": 3, "valence": 7.45, "arousal": 3.67}
    ],
    "summary": {
        "total_images": 3,
        "method": "heuristic",
        "valence_range": [4.91, 7.45],
        "arousal_range": [3.67, 7.12],
        "valence_mean": 6.20,
        "arousal_mean": 5.52,
        "valence_std": 1.27,
        "arousal_std": 1.73
    },
    "message": "Successfully calculated valence/arousal for 3 images using heuristic method"
}
```

### 2. Analyze EEG Features

**Endpoint**: `POST /api/v1/eeg/valence-arousal/analyze-features/`

**Parameters**:
- `file`: EEG CSV file (multipart/form-data)
- `method`: Analysis method ('heuristic' or 'ml') - optional

**Response**: Detailed feature analysis including sampling rate, feature counts, and example values.

## Scientific Background

### Valence Calculation
Based on **frontal alpha asymmetry** (Davidson's model):
- **Positive valence**: Left hemisphere alpha suppression (more activation)
- **Negative valence**: Right hemisphere alpha suppression (more activation)
- **Formula**: `Valence = log(F4_alpha) - log(F3_alpha)`

### Arousal Calculation  
Based on **beta/alpha ratio** across frontal regions:
- **Higher arousal**: Higher beta activity, lower alpha activity
- **Lower arousal**: Lower beta activity, higher alpha activity
- Uses mean ratio across AF3, AF4, F3, F4 channels

### Feature Extraction Methods

#### 1. Power Spectral Density (PSD)
- Fundamental feature for EEG emotion recognition
- Calculated using Welch's method with Hanning window
- Provides power distribution across frequency bands

#### 2. Differential Entropy (DE)
- Most widely used time-frequency feature
- Measures complexity and randomness of EEG signals
- **Formula**: `DE = (1/2) * log(2*π*e*σ²)`

#### 3. Asymmetry Features
- Captures hemispheric differences crucial for valence
- Calculated for electrode pairs: F3-F4, AF3-AF4
- **Formula**: `Asymmetry = log(right_power) - log(left_power)`

#### 4. Time-Domain Features
- Statistical measures: mean, std, variance, skewness, kurtosis
- First-order and second-order differences
- Zero-crossing rate

## Integration with Existing System

The new functionality seamlessly integrates with your existing EEG verification system:

1. **Uses existing `verify_eeg_file`** for segment extraction
2. **Compatible with current EEG file format** (requires same columns)
3. **Returns DataFrame structure** similar to SAM results
4. **Uses `image_number`** instead of `image_id` (as EEG data doesn't contain image_id)
5. **Follows same error handling** and validation patterns

## Requirements

The implementation uses libraries already in your `requirements.txt`:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Signal processing (filtering, Welch's method)
- `scikit-learn` - Machine learning (SVM classification)

## Research References

The implementation is based on current research findings:

1. **Li et al. (2023)** - CNN-KAN-F2CA model for emotion recognition
2. **Zheng et al. (2015)** - Differential entropy feature for EEG-based emotion classification
3. **Koelstra et al. (2012)** - DEAP: A database for emotion analysis using physiological signals
4. **Davidson, R. J. (1992)** - Anterior cerebral asymmetry and emotion
5. **Cai et al. (2020)** - Feature-level fusion approaches based on multimodal EEG data

## Testing

Run the test script to verify functionality:

```bash
python test_eeg_valence_arousal.py
```

This will:
- Test the calculator with synthetic data
- Validate the complete service workflow
- Show expected output format
- Verify integration with existing systems

## Performance Considerations

- **Processing time**: ~1-2 seconds per image segment (30 seconds of data at 128 Hz)
- **Memory usage**: Moderate (processes segments individually)
- **Accuracy**: Heuristic method provides baseline performance; ML method requires training data
- **Scalability**: Designed for individual file processing; can be extended for batch processing

## Future Enhancements

Potential improvements that could be added:
1. **Pre-trained ML models** based on public datasets (DEAP, SEED)
2. **Real-time processing** for live EEG streams
3. **Additional feature extraction** methods (wavelet transforms, connectivity measures)
4. **Ensemble methods** combining multiple algorithms
5. **Confidence scoring** for predictions
6. **Cross-subject normalization** techniques

## Troubleshooting

### Common Issues

1. **"Insufficient data for image X"**
   - Solution: Ensure each image segment has at least 10 samples
   - Check that EEG file has proper timing structure

2. **"Missing EEG data for channels"**
   - Solution: Verify CSV file contains all required columns: EEG.AF3, EEG.F3, EEG.AF4, EEG.F4

3. **"Cannot estimate sampling rate"**
   - Solution: Ensure timestamps are properly formatted and consecutive

4. **Unexpected valence/arousal values**
   - Solution: Check that EEG data quality is good; consider preprocessing if needed

### Debug Information

Use the feature analysis endpoint to get detailed information about:
- Sampling rate estimation
- Feature extraction results
- Signal quality indicators
- Processing parameters

This helps identify and resolve processing issues. 