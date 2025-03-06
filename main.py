import cv2
import numpy as np
import os
import random

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    return dilated

def find_white(processed_image):
    c=0
    pimg = np.array(processed_image)
    for i in range(len(pimg)):
        for j in range(len(pimg[0])):
            if (pimg[i][j]!=0):
                c+=1
    return c

def find_large_cracks(image, min_crack_size=500):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_cracks = [cnt for cnt in contours if cv2.contourArea(cnt) > min_crack_size]
    return len(large_cracks) 

def threshold_calc(folder_path, sample_size=100):
    all_images = os.listdir(folder_path)
    sum=[]
    sampled_images = random.sample(all_images, min(sample_size, len(all_images)))
    for filename in sampled_images:
        image_path = os.path.join(folder_path, filename)
        processed_image = preprocess_image(image_path)

  
        crack_count = find_large_cracks(processed_image)
        sum.append(crack_count)

    return sum



positive_folder = r"C:\Users\thero\Downloads\archive\Positive"
negative_folder = r"C:\Users\thero\Downloads\archive\Negative"

positive_counts = threshold_calc(positive_folder, sample_size=100)
negative_counts = threshold_calc(negative_folder, sample_size=100)
mainthresh=(np.mean(positive_counts)+np.mean(negative_counts))//2
print("Threshold for classification:", mainthresh)


def test_accuracy(folder_path, label, sample_size=100):
    all_images = os.listdir(folder_path)
    sampled_images = random.sample(all_images, min(sample_size, len(all_images)))

    correct = 0
    total = len(sampled_images)

    for filename in sampled_images:
        image_path = os.path.join(folder_path, filename)
        processed_image = preprocess_image(image_path)
        crack_count = find_large_cracks(processed_image)  

        prediction = "Positive" if crack_count > mainthresh else "Negative"

        if prediction == label:
            correct += 1

    return correct, total


correct_pos, total_pos = test_accuracy(positive_folder, "Positive", sample_size=1000)
correct_neg, total_neg = test_accuracy(negative_folder, "Negative", sample_size=1000)


total_correct = correct_pos + correct_neg
total_samples = total_pos + total_neg

accuracy = total_correct / total_samples
precision = correct_pos / (correct_pos + (total_neg - correct_neg))  
recall = correct_pos / total_pos 
f1_score = 2 * (precision * recall) / (precision + recall)


print(f"Total Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1_score*100:.2f}%")


def test_single_image(image_path, threshold):
    processed_image = preprocess_image(image_path)
    crack_count = find_large_cracks(processed_image)
    
    prediction = "Positive" if crack_count > threshold else "Negative"
    
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Crack Count: {crack_count}")
    print(f"Prediction: {prediction}")

test_single_image("surface_no_crack_test.png",mainthresh)
