try:
    from torchvision.models.vgg import model_urls
    print("Found in torchvision.models.vgg")
except ImportError as e:
    print(f"Not in torchvision.models.vgg: {e}")

try:
    from torchvision.models._api import model_urls
    print("Found in torchvision.models._api")
except ImportError as e:
    print(f"Not in torchvision.models._api: {e}")

try:
    from torchvision.models.utils import model_urls
    print("Found in torchvision.models.utils")
except ImportError as e:
    print(f"Not in torchvision.models.utils: {e}")

# Check what's available
import torchvision.models
print("Available in torchvision.models:", dir(torchvision.models))
