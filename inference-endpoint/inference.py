from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T


def main():
    run_cat_indentifier("cat_example_7.jpg")


def run_cat_indentifier(img_path):
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(img_path)

    # Pull pre-trained object detection model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = object_detection_model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {object_detection_model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

    labels = [object_detection_model.config.id2label.get(key) for key in results["labels"].tolist()]
    input_transform = T.Compose([ 
        T.PILToTensor() 
    ]) 
    img_tensor = input_transform(image)

    processed_image = torchvision.utils.draw_bounding_boxes(img_tensor, results["boxes"], labels)
    output_transform = T.ToPILImage()
    out_img = output_transform(processed_image)
    # out_img.save("out.jpg")

    # Crop images of each cat detected
    box_list = results["boxes"].tolist()
    cat_image_list = []

    for i, label in enumerate(labels):
        if label == "cat":
            cat_image = image.crop(box_list[i])
            cat_image_list.append(cat_image)
            cat_image.save(f'cat_{i}.jpg')
    
    # Run Cat Breed classifier
    classifier = pipeline("image-classification", model="checkpoint-16815")
    output_list = [classifier(cat_image)[0] for cat_image in cat_image_list]
    return {
        "cat_images": cat_image_list,
        "predicted_labels": [output['label'] for output in output_list]
    }


if __name__ == "__main__":
    main()