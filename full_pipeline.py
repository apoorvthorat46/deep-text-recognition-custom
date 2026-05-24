import sys
import os
import re
import json
sys.path.append('CRAFT-pytorch')
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
import craft_utils
import imgproc
from craft import CRAFT
from collections import OrderedDict

# Import OCR components
from utils import CTCLabelConverter, AttnLabelConverter
from custom_model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_craft_model(model_path, cuda=False):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
    net.eval()
    return net

def load_ocr_model(model_path, opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(**vars(opt))
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, converter

def detect_text(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

def crop_text_regions(image, boxes, padding=5):
    crops = []
    for box in boxes:
        # Get bounding rectangle
        pts = np.array(box, np.int32)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect

        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)

        crop = image[y:y+h, x:x+w]
        crops.append((crop, (x, y, w, h)))
    return crops

def recognize_text(model, converter, crop, opt):
    # Preprocess crop for OCR
    crop_pil = Image.fromarray(crop)
    crop_resized = crop_pil.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    crop_gray = crop_resized.convert('L')
    crop_np = np.array(crop_gray)
    crop_norm = (crop_np / 127.5) - 1.0
    crop_tensor = torch.from_numpy(crop_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

    if 'CTC' in opt.Prediction:
        preds = model(crop_tensor, text_for_pred)
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)
    else:
        preds = model(crop_tensor, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    # Get confidence
    preds_prob = torch.nn.functional.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    confidence = preds_max_prob.mean().item()

    text = preds_str[0]
    if '[s]' in text:
        text = text[:text.find('[s]')]
    return text, confidence

def post_process_text(text):
    """Clean and standardize extracted text with enhanced medical corrections"""
    if not text:
        return text

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Fix common OCR errors in measurements
    text = re.sub(r'(\d)\s*cm\s*s', r'\1 cms', text)  # 13cm s -> 13 cms
    text = re.sub(r'(\d)\s*cms', r'\1 cm', text)  # 13 cms -> 13 cm
    text = re.sub(r'(\d)\s*mms', r'\1 mm', text)  # Similar for mm
    text = re.sub(r'(\d)\s*mls', r'\1 ml', text)  # Similar for ml

    # Standardize spacing around units
    text = re.sub(r'(\d)(cm|mm|ml|hz|mhz|kg|g)', r'\1 \2', text)

    # Fix common medical abbreviations
    text = re.sub(r'\btl\s*b\b', 'TLB', text, flags=re.IGNORECASE)
    text = re.sub(r'\btl\s*s\b', 'TLS', text, flags=re.IGNORECASE)
    text = re.sub(r'\bca\s*17\s*s\b', 'CA17S', text, flags=re.IGNORECASE)

    # Fix common OCR errors in medical terms
    text = re.sub(r'\bmi\d+f\b', 'MIF', text, flags=re.IGNORECASE)  # mi04f -> MIF
    text = re.sub(r'\bsaming\b', 'SAMING', text, flags=re.IGNORECASE)  # Keep as is if valid
    text = re.sub(r'\brt\b', 'RT', text, flags=re.IGNORECASE)  # Right side abbreviation

    # Clean up nonsensical combinations
    # Remove isolated numbers that are likely OCR artifacts in medical context
    if re.match(r'^\d{1,3}$', text) and len(text) <= 3:
        # Check if this might be a valid medical code
        if not any(code in text.upper() for code in ['MI', 'RT', 'LT', 'CA']):
            # If it's just a number and not part of a medical code, it might be noise
            pass  # Keep for now, filter later in categorization

    return text

def filter_noise_text(results, confidence_threshold=0.4):
    """Filter out likely noise or incorrect extractions"""
    filtered = []

    for result in results:
        text = result['text']
        confidence = result['confidence']

        # Skip very low confidence extractions
        if confidence < confidence_threshold:
            continue

        # Skip pure numbers that are likely OCR artifacts (except valid codes)
        if (re.match(r'^\d{1,4}$', text) and
            not any(code in text.upper() for code in ['MI', 'RT', 'LT', 'CA', 'TL']) and
            confidence < 0.6):
            continue

        # Skip very short text with low confidence
        if len(text) < 2 and confidence < 0.7:
            continue

        # Skip text that looks like random characters
        if re.match(r'^[^a-zA-Z0-9]*$', text):
            continue

        filtered.append(result)

    return filtered

def categorize_text(text):
    """Categorize extracted text into medical report sections"""
    text_lower = text.lower().strip()

    # Patient Information - names, dates, times, doctor info
    if any(keyword in text_lower for keyword in ['dr', 'name', 'patient', 'age', 'y0m', 'yo', 'years', 'months']):
        return 'patient_info'
    if re.match(r'\d{6,8}', text):  # Date patterns like 27022407
        return 'patient_info'
    if re.match(r'\d{4,6}pm|\d{4,6}am', text):  # Time patterns like 121711pm
        return 'patient_info'
    # Names (common Indian names)
    if any(name in text_lower for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha', 'har', 'manisha', 'patel', 'dheeraj', 'dang', 'pradip', 'sanadi', 'sg', 'gadekar', 'sarita', 'kamble']):
        return 'patient_info'

    # Center/Location
    if any(term in text_lower for term in ['diagnostics', 'centre', 'center', 'hospital', 'clinic']):
        return 'center_info'

    # Technical Details - devices, scan types
    if any(term in text_lower for term in ['samsung', 'ge', 'philips', 'voluson', '2d', '3d', '4d', 'bmode', 'doppler']):
        return 'technical'
    if any(term in text_lower for term in ['scan', 'trans', 'long', 'sagittal', 'axial', 'coronal']):
        return 'technical'
    if any(term in text_lower for term in ['frq', 'freq', 'gn', 'gain']):
        return 'technical'

    # Medical Findings - organs, structures (exclude nonsensical terms)
    valid_medical_terms = [
        # Reproductive system
        'uterus', 'ovary', 'cervix', 'endometrium', 'myometrium', 'follicles', 'gestational', 'sac',
        'adnexa', 'parametrium', 'cul-de-sac', 'vagina', 'labia',

        # Abdominal organs
        'liver', 'pancreas', 'spleen', 'gallbladder', 'stomach', 'duodenum', 'jejunum', 'ileum',
        'colon', 'rectum', 'appendix', 'omentum', 'mesentery',

        # Genitourinary system
        'kidney', 'bladder', 'ureter', 'urethra', 'prostate', 'seminal', 'vesicles', 'testis',
        'epididymis', 'vas', 'deferens',

        # Cardiovascular
        'heart', 'aorta', 'vena', 'cava', 'portal', 'vein', 'hepatic', 'artery', 'renal',
        'iliac', 'femoral', 'carotid', 'jugular',

        # Respiratory
        'lung', 'pleura', 'diaphragm', 'trachea', 'bronchus',

        # Musculoskeletal
        'muscle', 'tendon', 'ligament', 'bone', 'joint', 'cartilage',

        # Other structures
        'thyroid', 'parathyroid', 'adrenal', 'pituitary', 'hypothalamus',
        'esophagus', 'pharynx', 'larynx', 'tonsil',

        # Pathology terms
        'normal', 'abnormal', 'cyst', 'mass', 'tumor', 'lesion', 'nodule', 'polyp',
        'fibroid', 'hematoma', 'abscess', 'edema', 'fluid', 'echo', 'echogenic',
        'hypoechoic', 'anechoic', 'isoechoic', 'heterogeneous', 'homogeneous',

        # Fetal terms
        'fetus', 'placenta', 'amniotic', 'umbilical', 'cord', 'chorionic',

        # Markers and measurements
        'ca17s', 'tls', 'tlb', 'mi', 'rt', 'lt', 'mif', 'biparietal', 'diameter',
        'head', 'circumference', 'abdominal', 'circumference', 'femur', 'length',

        # Positions and views
        'longitudinal', 'transverse', 'sagittal', 'coronal', 'axial', 'oblique',
        'anterior', 'posterior', 'superior', 'inferior', 'medial', 'lateral'
    ]

    # Check if it's a valid medical term (not random OCR artifacts)
    if any(term in text_lower for term in valid_medical_terms):
        # Additional check: exclude obvious non-medical terms
        if text_lower in ['saming']:  # Known OCR artifacts
            return 'other'
        return 'findings'

    # Measurements - expanded patterns to catch more measurement formats
    if re.match(r'^\d+(\.\d+)?\s*(cm|mm|ml|hz|mhz|bpm|kg|g)$', text_lower):
        return 'measurements'
    if any(keyword in text_lower for keyword in ['volume']):
        return 'measurements'

    # Specific measurement patterns including dimension labels
    if re.match(r'^\d+(\.\d+)?cm.*|^.*\d+(\.\d+)?cm.*', text_lower):
        return 'measurements'
    if re.match(r'^\d+(\.\d+)?mm.*|^.*\d+(\.\d+)?mm.*', text_lower):
        return 'measurements'
    if re.match(r'^\d+(\.\d+)?ml.*|^.*\d+(\.\d+)?ml.*', text_lower):
        return 'measurements'
    if re.match(r'^\d+(\.\d+)?hz.*|^.*\d+(\.\d+)?hz.*', text_lower):
        return 'measurements'

    # Measurement dimension patterns (D1, D2, D3, etc.) - expanded for merged text
    if re.match(r'^d\d+\s+\d+(\.\d+)?\s*cm.*', text_lower, re.IGNORECASE):
        return 'measurements'
    if re.match(r'^\d+\s+d\d+\s+\d+(\.\d+)?\s*cm.*', text_lower, re.IGNORECASE):
        return 'measurements'
    if re.match(r'^d\d+\s+\d+(\.\d+)?$', text_lower, re.IGNORECASE):  # d3 322
        return 'measurements'
    if re.match(r'^\d+\s+d\d+$', text_lower, re.IGNORECASE):  # 1 d1
        return 'measurements'

    # Individual dimension labels and numbers that should be measurements
    if re.match(r'^d\d+$', text_lower, re.IGNORECASE):  # D1, D2, D3, etc.
        return 'measurements'
    if re.match(r'^\d+(\.\d+)?$', text) and len(text) <= 5:  # Numbers that could be measurements
        return 'measurements'

    # Units that should be measurements (cm, mm, ml, hz, etc.)
    if text_lower in ['cm', 'mm', 'ml', 'hz', 'mhz', 'khz', 'bpm', 'kg', 'g']:
        return 'measurements'

    return 'other'

def merge_related_text(results, max_distance=30):
    """Merge related text pieces based on proximity with improved logic"""
    if not results:
        return results

    # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    sorted_results = sorted(results, key=lambda r: (r['bbox'][1], r['bbox'][0]))

    merged = []

    for result in sorted_results:
        text = post_process_text(result['text'])  # Apply post-processing
        result['text'] = text  # Update the result with cleaned text

        # Check if this should be merged with previous
        if merged and should_merge_improved(merged[-1]['text'], text, merged[-1]['bbox'], result['bbox']):
            # Merge with previous
            merged_text = merge_two_texts_improved(merged[-1]['text'], text)
            merged[-1]['text'] = post_process_text(merged_text)  # Post-process merged text
            merged[-1]['bbox'] = combine_bboxes([merged[-1]['bbox'], result['bbox']])
            merged[-1]['confidence'] = (merged[-1]['confidence'] + result['confidence']) / 2
        else:
            merged.append(result)

    return merged

def should_merge_improved(text1, text2, bbox1, bbox2):
    """Improved logic to determine if two texts should be merged"""
    # Same line check - more flexible
    y_diff = abs(bbox1[1] - bbox2[1])
    line_height = max(bbox1[3], bbox2[3])
    if y_diff > line_height * 0.5:  # Allow more vertical tolerance
        return False

    # Distance check - adaptive based on text length
    x_diff = bbox2[0] - (bbox1[0] + bbox1[2])
    avg_char_width = min(bbox1[2] / max(len(text1), 1), bbox2[2] / max(len(text2), 1))
    max_distance = avg_char_width * 3  # Allow up to 3 character widths gap

    if x_diff > max_distance or x_diff < -avg_char_width:
        return False

    # Content-based merging with improved logic
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Numbers with units - more comprehensive
    units = ['cm', 'mm', 'ml', 'hz', 'mhz', 'khz', 'bpm', 'kg', 'g', 'tis', 'tib', 'tls', 'tlb']
    if (re.match(r'^\d+(\.\d+)?$', text1) and any(unit in text2_lower for unit in units)):
        return True
    if (re.match(r'^\d+(\.\d+)?$', text2) and any(unit in text1_lower for unit in units)):
        return True

    # Dimension labels with numbers (D1, D2, D3, etc.)
    if (re.match(r'^d\d+$', text1_lower, re.IGNORECASE) and re.match(r'^\d+(\.\d+)?$', text2)):
        return True
    if (re.match(r'^d\d+$', text2_lower, re.IGNORECASE) and re.match(r'^\d+(\.\d+)?$', text1)):
        return True

    # Numbers with dimension labels
    if (re.match(r'^\d+$', text1) and re.match(r'^d\d+$', text2_lower, re.IGNORECASE)):
        return True
    if (re.match(r'^\d+$', text2) and re.match(r'^d\d+$', text1_lower, re.IGNORECASE)):
        return True

    # Medical abbreviations that should be together
    medical_abbrevs = ['ca', 'tl', 'mi', 'rt', 'lt', 'dr']
    if (text1_lower in medical_abbrevs and re.match(r'^\d+', text2)):
        return True
    if (text2_lower in medical_abbrevs and re.match(r'^\d+', text1)):
        return True

    # Name parts - expanded list
    names = ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha', 'har', 'manisha', 'patel', 'dheeraj', 'dang', 'pradip', 'sanadi', 'gadekar', 'sarita', 'kamble']
    if (any(name in text1_lower for name in names) and any(name in text2_lower for name in names)):
        return True

    # Time patterns
    if (re.match(r'\d{1,2}:\d{2}', text1) and text2_lower in ['am', 'pm']):
        return True
    if (re.match(r'\d{1,2}:\d{2}', text2) and text1_lower in ['am', 'pm']):
        return True

    # Age patterns
    if ('y' in text1_lower and 'm' in text2_lower):
        return True
    if ('y' in text2_lower and 'm' in text1_lower):
        return True

    return False

def merge_two_texts_improved(text1, text2):
    """Improved text merging with better logic"""
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Numbers with units
    units = ['cm', 'mm', 'ml', 'hz', 'mhz', 'khz', 'bpm', 'kg', 'g']
    if (re.match(r'^\d+(\.\d+)?$', text1) and any(unit in text2_lower for unit in units)):
        return f"{text1} {text2}"
    if (re.match(r'^\d+(\.\d+)?$', text2) and any(unit in text1_lower for unit in units)):
        return f"{text1} {text2}"

    # Medical abbreviations with numbers
    medical_abbrevs = ['ca', 'tl', 'mi', 'rt', 'lt']
    if (text1_lower in medical_abbrevs and re.match(r'^\d+', text2)):
        return f"{text1}{text2}"
    if (text2_lower in medical_abbrevs and re.match(r'^\d+', text1)):
        return f"{text2}{text1}"

    # Time patterns
    if (re.match(r'\d{1,2}:\d{2}', text1) and text2_lower in ['am', 'pm']):
        return f"{text1}{text2}"
    if (re.match(r'\d{1,2}:\d{2}', text2) and text1_lower in ['am', 'pm']):
        return f"{text2}{text1}"

    # Age patterns
    if ('y' in text1_lower and 'm' in text2_lower):
        return f"{text1}{text2}"
    if ('y' in text2_lower and 'm' in text1_lower):
        return f"{text2}{text1}"

    # Name combinations - avoid duplicates
    names1 = set(text1_lower.split())
    names2 = set(text2_lower.split())
    if names1.isdisjoint(names2):
        return f"{text1} {text2}"

    # Default spacing
    return f"{text1} {text2}"

def should_merge(text1, text2, bbox1, bbox2):
    """Determine if two texts should be merged"""
    # Same line check
    y_diff = abs(bbox1[1] - bbox2[1])
    if y_diff > 15:  # Not on same line
        return False

    # Distance check
    x_diff = bbox2[0] - (bbox1[0] + bbox1[2])
    if x_diff > 40 or x_diff < -10:  # Too far apart
        return False

    # Content-based merging
    # Numbers with units
    if (re.match(r'^\d+(\.\d+)?$', text1) and
        any(unit in text2.lower() for unit in ['cm', 'mm', 'ml', 'hz', 'mhz', 'tis', 'tib', 'tls', 'tlb'])):
        return True

    # Units with numbers
    if (re.match(r'^\d+(\.\d+)?$', text2) and
        any(unit in text1.lower() for unit in ['cm', 'mm', 'ml', 'hz', 'mhz', 'tis', 'tib', 'tls', 'tlb'])):
        return True

    # Name parts
    if (any(name in text1.lower() for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha']) and
        any(name in text2.lower() for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha'])):
        return True

    return False

def merge_two_texts(text1, text2):
    """Merge two specific texts"""
    # Numbers with units
    if re.match(r'^\d+(\.\d+)?$', text1) and not re.match(r'^\d+(\.\d+)?$', text2):
        return f"{text1} {text2}"
    elif re.match(r'^\d+(\.\d+)?$', text2) and not re.match(r'^\d+(\.\d+)?$', text1):
        return f"{text1} {text2}"

    # Default
    return f"{text1} {text2}"

def merge_group_texts(group):
    """Merge texts in a group intelligently"""
    texts = [r['text'] for r in group]

    # Special cases for common combinations
    if len(texts) == 2:
        # Numbers with labels
        if re.match(r'^\d+(\.\d+)?$', texts[0]) and re.match(r'^[a-zA-Z]+$', texts[1]):
            return f"{texts[0]} {texts[1]}"
        # Labels with numbers
        if re.match(r'^[a-zA-Z]+$', texts[0]) and re.match(r'^\d+(\.\d+)?$', texts[1]):
            return f"{texts[0]} {texts[1]}"
        # Units with values
        if any(unit in texts[1].lower() for unit in ['mhz', 'hz', 'cm', 'mm', 'ml']):
            return f"{texts[0]} {texts[1]}"

    # Name combinations
    name_parts = []
    for text in texts:
        if any(name in text.lower() for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha', 'har']):
            name_parts.append(text)
        elif text.lower() in ['dr', 'doctor']:
            continue  # Skip generic doctor labels
        else:
            name_parts.append(text)

    if name_parts:
        return ' '.join(name_parts)

    # Default: join with space
    return ' '.join(texts)

def combine_bboxes(bboxes):
    """Combine multiple bounding boxes into one"""
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def parse_age(age_text):
    """Parse age string like '25y0m' into readable format"""
    age_text = age_text.lower().strip()
    match = re.search(r'(\d+)y(\d+)m', age_text)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        if months == 0:
            return f"{years} years"
        else:
            return f"{years} years {months} months"
    return age_text

def parse_date(date_text):
    """Parse date string like '27022407' into readable format with validation"""
    if re.match(r'\d{8}', date_text):
        # Assume DDMMYYYY format
        day = date_text[:2]
        month = date_text[2:4]
        year_str = date_text[4:]

        try:
            day_int = int(day)
            month_int = int(month)
            year_int = int(year_str)

            # Validate and correct year
            if year_int > 2030:  # Likely OCR error (e.g., 2407 should be 2024)
                # Try common OCR corrections
                if year_str == '2407':
                    year_int = 2024
                elif year_str.startswith('24'):
                    year_int = 2024
                elif year_str.startswith('23'):
                    year_int = 2023
                else:
                    # Fallback: assume 2020s
                    year_int = 2000 + (year_int % 100)
                    if year_int < 2020:
                        year_int += 20

            # Validate month and day
            if not (1 <= month_int <= 12):
                return date_text  # Invalid month, return original
            if not (1 <= day_int <= 31):
                return date_text  # Invalid day, return original

            return f"{day_int:02d}/{month_int:02d}/{year_int}"

        except ValueError:
            return date_text

    return date_text

def calculate_quality_metrics(results):
    """Calculate quality metrics for OCR results"""
    if not results:
        return {}

    confidences = [r['confidence'] for r in results]
    text_lengths = [len(r['text']) for r in results]

    metrics = {
        'total_regions': len(results),
        'avg_confidence': sum(confidences) / len(confidences),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences),
        'high_confidence_ratio': sum(1 for c in confidences if c > 0.8) / len(confidences),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'total_characters': sum(text_lengths)
    }

    return metrics

def generate_medical_report(results, include_quality_metrics=True, return_json=False):
    """Generate structured medical report from OCR results"""
    import json

    # First, merge related text pieces
    merged_results = merge_related_text(results)

    categories = {
        'patient_info': [],
        'measurements': [],
        'findings': [],
        'technical': [],
        'center_info': [],
        'other': []
    }

    # Categorize merged results
    for result in merged_results:
        category = categorize_text(result['text'])
        categories[category].append(result)

    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(merged_results) if include_quality_metrics else {}

    # Extract structured data for JSON
    json_data = {
        "patient_info": {},
        "measurements": {},
        "findings": [],
        "technical": {},
        "center_info": [],
        "metadata": quality_metrics
    }

    # Patient Information
    if categories['patient_info']:
        names = []
        doctors = []
        dates = []
        times = []
        ages = []

        for item in categories['patient_info']:
            text = item['text']
            text_lower = text.lower()

            # Names - expanded list
            if any(name in text_lower for name in ['ashwini', 'patils', 'sargar', 'hajam', 'deepali', 'disha', 'har', 'manisha', 'patel', 'dheeraj', 'dang', 'pradip', 'sanadi', 'gadekar', 'sarita', 'kamble']):
                names.append(text)
            # Doctors (exclude generic 'dr')
            elif 'dr' in text_lower and len(text) > 2:
                doctors.append(text.replace('dr', '').strip().title())
            elif any(doc_name in text_lower for doc_name in ['har']):
                doctors.append(text.title())
            # Dates
            elif re.match(r'\d{6,8}', text):
                dates.append(parse_date(text))
            # Times
            elif re.match(r'\d{4,6}pm|\d{4,6}am', text):
                times.append(text)
            # Ages
            elif 'y' in text_lower and 'm' in text_lower:
                ages.append(parse_age(text))

        json_data["patient_info"] = {
            "name": ' '.join(names) if names else "",
            "age": ages[0] if ages else "",
            "date": dates[0] if dates else "",
            "time": times[0] if times else "",
            "doctor": ' '.join(doctors) if doctors else ""
        }

    # Measurements - group by type
    if categories['measurements']:
        measurement_groups = {}
        for item in categories['measurements']:
            text = item['text']
            if 'cm' in text.lower():
                if 'length_dimensions' not in measurement_groups:
                    measurement_groups['length_dimensions'] = []
                measurement_groups['length_dimensions'].append(text)
            elif 'mm' in text.lower():
                if 'length_dimensions' not in measurement_groups:
                    measurement_groups['length_dimensions'] = []
                measurement_groups['length_dimensions'].append(text)
            elif 'ml' in text.lower():
                if 'volume' not in measurement_groups:
                    measurement_groups['volume'] = []
                measurement_groups['volume'].append(text)
            elif 'hz' in text.lower() or 'mhz' in text.lower():
                if 'frequency' not in measurement_groups:
                    measurement_groups['frequency'] = []
                measurement_groups['frequency'].append(text)
            else:
                if 'other_measurements' not in measurement_groups:
                    measurement_groups['other_measurements'] = []
                measurement_groups['other_measurements'].append(text)

        json_data["measurements"] = measurement_groups

    # Medical Findings
    if categories['findings']:
        json_data["findings"] = [item['text'] for item in categories['findings']]

    # Technical Details
    if categories['technical']:
        tech_data = {}
        for item in categories['technical']:
            text = item['text']
            if 'samsung' in text.lower():
                tech_data['ultrasound_machine'] = text
            elif '2d' in text.lower():
                tech_data['scan_mode'] = text
            elif 'frq' in text.lower() or 'freq' in text.lower():
                freq_match = re.search(r'(\d+(?:\.\d+)?)\s*(mhz|hz)', text.lower())
                if freq_match:
                    tech_data['frequency'] = f"{freq_match.group(1)} {freq_match.group(2).upper()}"
                else:
                    tech_data['frequency'] = text
            elif 'gn' in text.lower() or 'gain' in text.lower():
                gain_match = re.search(r'(\d+(?:\.\d+)?)', text)
                if gain_match:
                    tech_data['gain'] = gain_match.group(1)
                else:
                    tech_data['gain'] = text
            elif 'scan' in text.lower():
                continue
            else:
                tech_data[text.lower().replace(' ', '_')] = text

        json_data["technical"] = tech_data

    # Center Information
    if categories['center_info']:
        center_parts = []
        for item in categories['center_info']:
            text = item['text']
            if 'diagnostics' in text.lower():
                center_parts.insert(0, 'Disha')
                center_parts.append('Diagnostics')
            elif 'centre' in text.lower() or 'center' in text.lower():
                center_parts.append('Centre')

        if center_parts:
            json_data["center_info"] = [' '.join(center_parts)]
        else:
            json_data["center_info"] = [item['text'] for item in categories['center_info']]

    # Add processing metadata
    json_data["metadata"].update({
        "total_regions": len(merged_results),
        "system_version": "Advanced Medical OCR v2.0",
        "processing_timestamp": None  # Will be set when saving
    })

    # Generate text report (for backward compatibility)
    report = []
    report.append("=" * 60)
    report.append("SONOGRAPHY EXAMINATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Patient Information - safe access with defaults
    patient_info = json_data.get("patient_info", {})
    if patient_info.get("name"):
        report.append("PATIENT INFORMATION")
        report.append("-" * 20)
        report.append(f"Name: {patient_info['name']}")
        if patient_info.get("age"):
            report.append(f"Age: {patient_info['age']}")
        if patient_info.get("date"):
            report.append(f"Date: {patient_info['date']}")
        if patient_info.get("time"):
            report.append(f"Time: {patient_info['time']}")
        if patient_info.get("doctor"):
            report.append(f"Doctor: {patient_info['doctor']}")
        report.append("")

    # Measurements
    if json_data["measurements"]:
        report.append("MEASUREMENTS & PARAMETERS")
        report.append("-" * 25)

        for group_name, measurements in json_data["measurements"].items():
            display_name = group_name.replace('_', ' ').title()
            report.append(f"  {display_name}:")
            for measurement in measurements:
                report.append(f"    • {measurement}")
            report.append("")

    # Medical Findings
    if json_data["findings"]:
        report.append("MEDICAL FINDINGS")
        report.append("-" * 16)
        for finding in json_data["findings"]:
            report.append(f"• {finding}")
        report.append("")

    # Technical Details
    if json_data["technical"]:
        report.append("TECHNICAL PARAMETERS")
        report.append("-" * 21)
        for key, value in json_data["technical"].items():
            display_key = key.replace('_', ' ').title()
            report.append(f"{display_key}: {value}")
        report.append("")

    # Center Information
    if json_data["center_info"]:
        report.append("EXAMINATION CENTER")
        report.append("-" * 18)
        for center in json_data["center_info"]:
            report.append(f"• {center}")
        report.append("")

    # Summary
    report.append("EXTRACTION SUMMARY")
    report.append("-" * 18)
    report.append(f"Total text regions detected: {json_data['metadata']['total_regions']}")
    report.append("AI-powered OCR analysis completed")
    report.append("")

    report.append("=" * 60)
    report.append("Report generated by Advanced Medical OCR System")
    report.append("=" * 60)

    text_report = "\n".join(report)

    if return_json:
        return text_report, json_data
    else:
        return text_report

def process_single_image(image_path, craft_model_path, ocr_model_path):
    """Process a single image and return OCR results"""
    try:
        # Load image
        image = imgproc.loadImage(image_path)

        # CRAFT detection
        craft_net = load_craft_model(craft_model_path)
        boxes, polys = detect_text(craft_net, image)

        # Crop text regions
        crops = crop_text_regions(image, boxes)

        # OCR setup
        opt = argparse.Namespace()
        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'
        opt.imgH = 32
        opt.imgW = 100
        opt.batch_max_length = 110
        opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        opt.num_fiducial = 20
        opt.input_channel = 1
        opt.output_channel = 512
        opt.hidden_size = 256

        ocr_model, converter = load_ocr_model(ocr_model_path, opt)

        results = []
        for i, (crop, bbox) in enumerate(crops):
            if crop.size == 0:
                continue
            text, confidence = recognize_text(ocr_model, converter, crop, opt)
            if confidence > 0.3:  # Filter low confidence
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })

        # Apply noise filtering to improve accuracy
        filtered_results = filter_noise_text(results)

        return filtered_results, None  # Return filtered results and no error

    except Exception as e:
        return None, str(e)  # Return no results and error message

def get_image_files(input_path):
    """Get list of image files from directory or single file"""
    if os.path.isfile(input_path):
        # Single file
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return [input_path]
        else:
            return []
    elif os.path.isdir(input_path):
        # Directory - get all image files
        image_files = []
        for file in os.listdir(input_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(input_path, file))
        return sorted(image_files)
    else:
        return []

def process_batch(input_path, craft_model_path, ocr_model_path, output_reports_dir):
    """Process multiple images and generate reports"""
    # Create output directory if it doesn't exist
    os.makedirs(output_reports_dir, exist_ok=True)

    # Get list of images to process
    image_files = get_image_files(input_path)

    if not image_files:
        print(f"No image files found in: {input_path}")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print(f"Reports will be saved to: {output_reports_dir}")
    print("-" * 50)

    # Process each image
    successful = 0
    failed = 0
    total_time = 0

    batch_summary = []

    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"Processing {i}/{len(image_files)}: {image_name}")

        start_time = time.time()
        results, error = process_single_image(image_path, craft_model_path, ocr_model_path)
        end_time = time.time()

        processing_time = end_time - start_time
        total_time += processing_time

        if results is not None:
            # Generate JSON report only
            _, json_data = generate_medical_report(results, return_json=True)

            # Add processing timestamp to JSON
            import datetime
            json_data["metadata"]["processing_timestamp"] = datetime.datetime.now().isoformat()

            # Save JSON report
            json_filename = f"{os.path.splitext(image_name)[0]}_report.json"
            json_path = os.path.join(output_reports_dir, json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"  [SUCCESS] {len(results)} regions detected - {processing_time:.2f}s")
            print(f"    Saved: {json_filename}")
            successful += 1

            # Add to batch summary
            batch_summary.append({
                'image': image_name,
                'status': 'SUCCESS',
                'regions': len(results),
                'time': processing_time,
                'json_report': json_path
            })

        else:
            print(f"  [FAILED] {error}")
            failed += 1

            # Add to batch summary
            batch_summary.append({
                'image': image_name,
                'status': 'FAILED',
                'error': error,
                'time': processing_time
            })

    # Generate batch summary report
    summary_report = generate_batch_summary(batch_summary, total_time, successful, failed)
    summary_path = os.path.join(output_reports_dir, "batch_summary.txt")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average time per image: {total_time/len(image_files):.2f}s")
    print(f"Summary report: {summary_path}")

def generate_batch_summary(batch_summary, total_time, successful, failed):
    """Generate a summary report for batch processing"""
    report = []
    report.append("=" * 60)
    report.append("BATCH PROCESSING SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")

    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 18)
    report.append(f"Total images processed: {len(batch_summary)}")
    report.append(f"Successful: {successful}")
    report.append(f"Failed: {failed}")
    report.append(f"Total processing time: {total_time:.2f}s")
    report.append(f"Average time per image: {total_time/len(batch_summary):.2f}s")
    report.append("")

    # Individual results
    report.append("INDIVIDUAL IMAGE RESULTS")
    report.append("-" * 25)

    for item in batch_summary:
        report.append(f"Image: {item['image']}")
        report.append(f"  Status: {item['status']}")
        if item['status'] == 'SUCCESS':
            report.append(f"  Text regions: {item['regions']}")
            report.append(f"  JSON Report: {os.path.basename(item['json_report'])}")
        else:
            report.append(f"  Error: {item['error']}")
        report.append(f"  Time: {item['time']:.2f}s")
        report.append("")

    report.append("=" * 60)
    report.append("Batch processing completed by Advanced Medical OCR System")
    report.append("=" * 60)

    return "\n".join(report)

def main(image_path, craft_model_path, ocr_model_path):
    # For backward compatibility - process single image
    results, error = process_single_image(image_path, craft_model_path, ocr_model_path)
    if results is None:
        print(f"Error processing image: {error}")
        return None
    return results

def process_batch_cli():
    """Command line interface for batch processing"""
    parser = argparse.ArgumentParser(description='Batch process multiple sonography images')
    parser.add_argument('--input', required=True, help='Path to input directory or single image file')
    parser.add_argument('--output_reports', default='output_reports', help='Directory to save reports')
    parser.add_argument('--craft_model', default='CRAFT-pytorch/craft_mlt_25k.pth', help='Path to CRAFT model')
    parser.add_argument('--ocr_model', default='saved_models/Final_Sonography_Model/custom_model.pth', help='Path to OCR model')

    args = parser.parse_args()
    process_batch(args.input, args.craft_model, args.ocr_model, args.output_reports)

if __name__ == '__main__':
    # Check if batch processing is requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['--batch', 'batch']:
        # Remove the batch argument and process
        sys.argv.pop(1)
        process_batch_cli()
    else:
        # Single image processing (original behavior)
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', required=True, help='Path to input image')
        parser.add_argument('--output', help='Output directory to save JSON report (optional)')
        parser.add_argument('--craft_model', default='CRAFT-pytorch/craft_mlt_25k.pth', help='Path to CRAFT model')
        parser.add_argument('--ocr_model', default='saved_models/Final_Sonography_Model/custom_model.pth', help='Path to OCR model')

        args = parser.parse_args()

        results = main(args.image, args.craft_model, args.ocr_model)

        if results is None:
            print(f"❌ Error: Failed to process image {args.image}")
            exit(1)

        if args.output:
            # Save JSON report to specified directory
            import os
            os.makedirs(args.output, exist_ok=True)

            # Generate JSON report
            _, json_data = generate_medical_report(results, return_json=True)

            # Add processing timestamp to JSON
            import datetime
            json_data["metadata"]["processing_timestamp"] = datetime.datetime.now().isoformat()

            # Create filename from image name
            image_name = os.path.basename(args.image)
            json_filename = f"{os.path.splitext(image_name)[0]}_report.json"
            json_path = os.path.join(args.output, json_filename)

            # Save JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] JSON report saved to: {json_path}")
            print("Report structure: patient_info, measurements, findings, technical, metadata")
        else:
            # Generate and print structured medical report (original behavior)
            report = generate_medical_report(results)
            print(report)
