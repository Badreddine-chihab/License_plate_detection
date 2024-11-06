import os

def clean_dataset(images_dir, annotations_dir):
    # Get lists of image and annotation files
    image_files = {f for f in os.listdir(images_dir) if f.endswith('.jpg')}
    annotation_files = {f.replace('.xml', '.jpg') for f in os.listdir(annotations_dir) if f.endswith('.xml')}
    
    # Check for missing annotations and delete images
    for image_file in image_files:
        if image_file not in annotation_files:
            print(f"Deleting image {image_file} as it has no corresponding annotation.")
            os.remove(os.path.join(images_dir, image_file))
    
    # Check for missing images and delete annotations
    for annotation_file in annotation_files:
        if annotation_file not in image_files:
            print(f"Deleting annotation {annotation_file.replace('.jpg', '.xml')} as it has no corresponding image.")
            os.remove(os.path.join(annotations_dir, annotation_file.replace('.jpg', '.xml')))

if __name__ == "__main__":
    images_dir = './dataset/images'
    annotations_dir = './dataset/annotations'
    clean_dataset(images_dir, annotations_dir)
