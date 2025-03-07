from ultralytics import YOLO

def main():
    # Load the YOLOv11 model (nano variant). For improved performance, you may consider a larger model.
    model = YOLO("yolo11n.pt")
    
    try:
        # Train the model with individual hyperparameter overrides.
        # The 'project' and 'name' parameters set the output directory to:
        # C:\D_DRIVE\preprocessed2\mini1\train5
        train_results = model.train(
            data="C:/D_DRIVE/preprocessed2/brats_yolo.yaml",  # Path to your dataset YAML config.
            epochs=30,              # Increased epochs for better convergence.
            imgsz=640,               # Training image size.
            batch=16,                # Increase batch size if GPU memory allows.
            device="cuda",           # Use GPU.
            lr0=0.01,                # Initial learning rate.
            momentum=0.937,          # Momentum factor.
            weight_decay=0.0005,     # L2 regularization.
            warmup_epochs=3.0,       # Number of warmup epochs.
            box=7.5,                 # Box loss weight.
            cls=0.3,                 # Lower classification loss weight to encourage better boundary predictions.
            dfl=1.0,                 # Distribution focal loss weight.
            augment=True,            # Enable data augmentation.
            mosaic=1.0,              # Enable mosaic augmentation.
            mixup=0.0,               # Disable mixup augmentation.
            perspective=0.0,         # Disable perspective augmentation.
            project="C:/D_DRIVE/preprocessed2/mini1",  # Custom output base directory.
            name="train5",           # Name of the training run (subfolder created under the project folder).
            verbose=True             # Enable detailed logging.
        )
    except Exception as e:
        print("An error occurred during training:", e)
    
    try:
        # Evaluate the model on the validation set.
        # Ensure the data parameter points to your custom YAML file.
        metrics = model.val(imgsz=640, data="C:/D_DRIVE/preprocessed/brats_yolo.yaml")
        print("Validation metrics:", metrics)
    except Exception as e:
        print("An error occurred during evaluation:", e)
    
    try:
        # Update the path below to point to an existing test image.
        test_image = "C:/D_DRIVE/preprocessed/test/images/example.png"  # <-- Update this path
        results = model.predict(source=test_image, imgsz=640)
        results[0].show()  # Display the results
    except Exception as e:
        print("An error occurred during inference:", e)
    
    try:
        # Export the trained model to ONNX format.
        onnx_path = model.export(format="onnx")
        print("Exported ONNX model to:", onnx_path)
    except Exception as e:
        print("An error occurred during model export:", e)

if __name__ == '__main__':
    main()
