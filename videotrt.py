import os
import cv2
import torch
import time
from ultralytics import YOLO, settings

def optimizeModel(modelPath, precision='fp32'):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return YOLO(modelPath)

    enginePath = modelPath.replace('.pt', f'_{precision}.engine')

    if os.path.exists(enginePath):
        print(f"Found existing TensorRT engine: {enginePath}")
        return YOLO(enginePath)

    print(f"No existing engine found. Converting '{modelPath}' to TensorRT {precision.upper()}...")
    
    model = YOLO(modelPath)
    
    halfMode = (precision == 'fp16')
    int8Mode = (precision == 'int8')
    
    try:
        model.export(format='engine', half=halfMode, int8=int8Mode, device=0)
        print("Export successful. Loading the new engine file.")
        return YOLO(enginePath)
    except Exception as e:
        print(f"Error during TensorRT conversion: {e}")
        print("Reverting to the original PyTorch model.")
        return YOLO(modelPath)

def runInference(model, videoPath, modelName, display=True):
    print(f"\nProcessing video with '{modelName}'")
    
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{videoPath}'.")
        return 0.0

    frameCount = 0
    startTime = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frameCount += 1
        
        results = model(frame, conf=0.5, verbose=False)
        
        if display:
            annotatedFrame = results[0].plot()
            fps = frameCount / (time.time() - startTime)
            
            cv2.putText(annotatedFrame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotatedFrame, f"Model: {modelName}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Video Inference', annotatedFrame)
            
            if cv2.waitKey(1) & 0xFF == 27: #esc
                break

    endTime = time.time()
    totalTime = endTime - startTime
    
    cap.release()
    if display:
        cv2.destroyAllWindows()

    if totalTime > 0:
        avgFps = frameCount / totalTime
        print(f"Processed frames: {frameCount}")
        print(f"Total time: {totalTime:.2f} seconds")
        print(f"Average Frame Processing Speed (FPS): {avgFps:.2f}")
        return avgFps
    else:
        print("Could not calculate FPS.")
        return 0.0

if __name__ == '__main__':
    videoPath = 'main.mp4'
    modelPath = 'yolo11m.pt'

    print("\nChoose TensorRT precision:")
    print("1. FP32")
    print("2. FP16")
    print("3. INT8")
    precisionChoice = input("Enter choice (1-3): ").strip()

    precisionMap = {
        '1': 'fp32',
        '2': 'fp16',
        '3': 'int8'
    }

    precision = precisionMap.get(precisionChoice)
    if not precision:
        print("Invalid choice. Defaulting to FP32.")
        precision = 'fp32'

    originalModel = YOLO(modelPath)
    originalFps = runInference(originalModel, videoPath, "Original YOLOv11", display=True)
    
    optimizedModel = optimizeModel(modelPath, precision=precision)
    optimizedFps = runInference(optimizedModel, videoPath, f"TensorRT {precision.upper()} YOLOv11", display=True)
    
    print(f"\nPerformance Comparison with: {precision}")
    print(f"Original Model Average FPS: {originalFps:.2f}")
    print(f"Optimized Model Average FPS: {optimizedFps:.2f}")
    
    if originalFps > 0:
        speedUp = optimizedFps / originalFps
        print(f"Speed improvement: {speedUp:.2f}x faster")
    else:
        print("Speed-up ratio could not be calculated.")