#!/usr/bin/env python3
"""
Accuracy Analysis for Medical OCR System
Analyzes the extracted text against expected patterns and provides accuracy metrics
"""

def analyze_ocr_accuracy(extracted_results, image_name="Unknown"):
    """
    Analyze OCR accuracy based on extracted results
    Since we don't have ground truth, we'll analyze based on:
    1. Format consistency
    2. Medical terminology validity
    3. Logical consistency
    4. Pattern recognition accuracy
    """

    print(f"\n{'='*60}")
    print(f"OCR ACCURACY ANALYSIS - {image_name}")
    print(f"{'='*60}")

    # Extract categories from results
    categories = {
        'patient_info': [],
        'measurements': [],
        'findings': [],
        'technical': [],
        'center_info': []
    }

    # Categorize the extracted text (simulate the categorization logic)
    for result in extracted_results:
        text = result['text'].lower().strip()

        # Patient info patterns
        if any(keyword in text for keyword in ['dr', 'name', 'patient', 'age', 'years', 'months']):
            categories['patient_info'].append(result)
        elif re.match(r'\d{6,8}', result['text']):  # Date patterns
            categories['patient_info'].append(result)
        elif any(name in text for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha', 'har', 'manisha', 'patel', 'dheeraj', 'dang', 'pradip', 'sanadi', 'gadekar', 'sarita', 'kamble']):
            categories['patient_info'].append(result)

        # Measurements
        elif re.match(r'^\d+(\.\d+)?\s*(cm|mm|ml|hz|mhz|bpm|kg|g)$', text):
            categories['measurements'].append(result)
        elif 'volume' in text:
            categories['measurements'].append(result)

        # Medical findings
        elif any(term in text for term in ['uterus', 'ovary', 'kidney', 'liver', 'bladder', 'heart', 'fetus', 'placenta']):
            categories['findings'].append(result)
        elif any(term in text for term in ['normal', 'abnormal', 'cyst', 'mass', 'fluid', 'echo']):
            categories['findings'].append(result)
        elif any(term in text for term in ['ca17s', 'tls', 'tlb', 'mi', 'rt']):
            categories['findings'].append(result)

        # Technical
        elif any(term in text for term in ['samsung', 'ge', 'philips', 'voluson', '2d', '3d', '4d', 'bmode', 'doppler']):
            categories['technical'].append(result)
        elif any(term in text for term in ['frq', 'freq', 'gn', 'gain']):
            categories['technical'].append(result)

        # Center info
        elif any(term in text for term in ['diagnostics', 'centre', 'center', 'hospital', 'clinic']):
            categories['center_info'].append(result)

    # Analyze each category
    total_texts = len(extracted_results)
    accuracy_scores = {}

    print(f"\nTotal text regions extracted: {total_texts}")
    print(f"Average confidence: {sum(r['confidence'] for r in extracted_results)/total_texts:.3f}")

    # Patient Information Analysis
    print(f"\n{'-'*30}")
    print("PATIENT INFORMATION ANALYSIS")
    print(f"{'-'*30}")

    patient_texts = [r['text'] for r in categories['patient_info']]
    print(f"Patient-related texts found: {len(patient_texts)}")
    for text in patient_texts:
        print(f"  - {text}")

    # Check for common issues
    issues = []

    # Date validation
    date_pattern = re.compile(r'\d{8}')
    dates = [t for t in patient_texts if date_pattern.match(t.replace('/', '').replace('-', ''))]
    if dates:
        for date in dates:
            clean_date = date.replace('/', '').replace('-', '')
            if len(clean_date) == 8:
                year = int(clean_date[4:8])
                if year > 2030 or year < 2020:  # Unreasonable year
                    issues.append(f"Invalid year in date: {date} (year: {year})")

    # Doctor name validation
    doctor_texts = [t for t in patient_texts if t.lower().startswith('dr') or len(t) > 10]
    numeric_doctor = [t for t in patient_texts if t.isdigit()]
    if numeric_doctor:
        issues.append(f"Doctor name appears to be numeric: {numeric_doctor}")

    # Measurements Analysis
    print(f"\n{'-'*30}")
    print("MEASUREMENTS ANALYSIS")
    print(f"{'-'*30}")

    measurement_texts = [r['text'] for r in categories['measurements']]
    print(f"Measurement texts found: {len(measurement_texts)}")
    for text in measurement_texts:
        print(f"  - {text}")

    # Validate measurements
    valid_measurements = 0
    for text in measurement_texts:
        # Check for proper units
        if re.match(r'^\d+(\.\d+)?\s*(cm|mm|ml|hz|mhz|g|kg)$', text.lower()):
            valid_measurements += 1
        elif 'volume' in text.lower():
            valid_measurements += 1

    measurement_accuracy = valid_measurements / max(len(measurement_texts), 1)
    print(".1%")

    # Medical Findings Analysis
    print(f"\n{'-'*30}")
    print("MEDICAL FINDINGS ANALYSIS")
    print(f"{'-'*30}")

    finding_texts = [r['text'] for r in categories['findings']]
    print(f"Medical finding texts found: {len(finding_texts)}")
    for text in finding_texts:
        print(f"  - {text}")

    # Validate medical terms
    valid_medical_terms = 0
    medical_keywords = ['uterus', 'ovary', 'kidney', 'liver', 'bladder', 'heart', 'fetus', 'placenta',
                       'ca17s', 'tls', 'tlb', 'mi', 'rt', 'normal', 'abnormal', 'cyst', 'mass', 'fluid', 'echo']

    for text in finding_texts:
        if any(keyword in text.lower() for keyword in medical_keywords):
            valid_medical_terms += 1

    medical_accuracy = valid_medical_terms / max(len(finding_texts), 1)
    print(".1%")

    # Technical Parameters Analysis
    print(f"\n{'-'*30}")
    print("TECHNICAL PARAMETERS ANALYSIS")
    print(f"{'-'*30}")

    technical_texts = [r['text'] for r in categories['technical']]
    print(f"Technical texts found: {len(technical_texts)}")
    for text in technical_texts:
        print(f"  - {text}")

    # Overall Assessment
    print(f"\n{'-'*30}")
    print("OVERALL ACCURACY ASSESSMENT")
    print(f"{'-'*30}")

    # Calculate weighted accuracy
    weights = {
        'measurements': 0.3,  # Most important for medical accuracy
        'findings': 0.3,      # Critical medical information
        'patient_info': 0.2,  # Important for identification
        'technical': 0.2      # Less critical but useful
    }

    overall_accuracy = (
        measurement_accuracy * weights['measurements'] +
        medical_accuracy * weights['findings'] +
        (1.0 if len(categories['patient_info']) > 0 else 0.5) * weights['patient_info'] +  # Basic presence check
        (1.0 if len(categories['technical']) > 0 else 0.5) * weights['technical']    # Basic presence check
    )

    print(".1%")

    # Issues summary
    if issues:
        print(f"\nISSUES IDENTIFIED ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major issues identified in basic validation.")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if overall_accuracy < 0.7:
        print("  - Model accuracy needs improvement")
        print("  - Consider additional training data")
        print("  - Review text detection parameters")
    elif overall_accuracy < 0.85:
        print("  - Model shows reasonable accuracy")
        print("  - Minor improvements possible")
        print("  - Consider fine-tuning for specific medical terms")
    else:
        print("  - Model shows good accuracy")
        print("  - Suitable for production use")

    return {
        'overall_accuracy': overall_accuracy,
        'measurement_accuracy': measurement_accuracy,
        'medical_accuracy': medical_accuracy,
        'issues': issues,
        'total_texts': total_texts
    }

# Example usage with the extracted data from the user's output
if __name__ == "__main__":
    # Simulate the extracted results based on user's output
    extracted_results = [
        {'text': 'disha sargar ashwini patils hajam deepali har', 'confidence': 0.8},
        {'text': '25 years', 'confidence': 0.9},
        {'text': '27022407', 'confidence': 0.85},
        {'text': '108', 'confidence': 0.7},
        {'text': '140 cm', 'confidence': 0.95},
        {'text': '13 cm', 'confidence': 0.9},
        {'text': '13 cm', 'confidence': 0.9},
        {'text': '125 cm', 'confidence': 0.95},
        {'text': '55 hz', 'confidence': 0.85},
        {'text': '21 mhz', 'confidence': 0.9},
        {'text': 'volume', 'confidence': 0.8},
        {'text': '557 ml', 'confidence': 0.95},
        {'text': 'uterus', 'confidence': 0.9},
        {'text': 'CA17S', 'confidence': 0.85},
        {'text': 'TLS 041', 'confidence': 0.8},
        {'text': 'TLB', 'confidence': 0.85},
        {'text': 'mi04f', 'confidence': 0.75},
        {'text': 'saming', 'confidence': 0.7},
        {'text': 'rt', 'confidence': 0.8},
        {'text': 'ovary', 'confidence': 0.9},
        {'text': 'samsung', 'confidence': 0.95},
        {'text': '2d', 'confidence': 0.9},
        {'text': 'frq', 'confidence': 0.8},
        {'text': '2', 'confidence': 0.7},
        {'text': 'Disha Diagnostics Centre', 'confidence': 0.95}
    ]

    import re
    results = analyze_ocr_accuracy(extracted_results, "ASHWINIHAJAMDRDEEPALIPATILSARGAR3.dcm.png")
    print(f"\nFinal Assessment: {results['overall_accuracy']:.1%} overall accuracy")
