import argparse
import torch
from PIL import Image
import numpy as np

# Import OCR components
from utils import CTCLabelConverter, AttnLabelConverter
from custom_model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
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

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(**vars(opt))
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, converter, opt

def preprocess_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((100, 32), Image.BICUBIC)  # Resize to model input size
    image = np.array(image)
    image = (image / 127.5) - 1.0  # Normalize
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
    return image

def recognize_text(model, converter, image_tensor, opt):
    length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

    with torch.no_grad():
        if 'CTC' in opt.Prediction:
            preds = model(image_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(image_tensor, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

    # Calculate confidence
    preds_prob = torch.nn.functional.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    confidence = preds_max_prob.mean().item()

    text = preds_str[0]
    if '[s]' in text:
        text = text[:text.find('[s]')]
    return text, confidence

def main():
    parser = argparse.ArgumentParser(description='Simple OCR using Fine-tuned Sonography Model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='saved_models/Final_Sonography_Model/custom_model.pth',
                       help='Path to model file')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, converter, opt = load_model(args.model)

    # Preprocess image
    print("Preprocessing image...")
    image_tensor = preprocess_image(args.image)

    # Recognize text
    print("Recognizing text...")
    text, confidence = recognize_text(model, converter, image_tensor, opt)

    print(f"\nResult:")
    print(f"Text: '{text}'")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()
