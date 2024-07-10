import xml.etree.ElementTree
import torch
import torch.optim

import create_dataset
import model
import process_data


xml_train = xml.etree.ElementTree.parse('resources/labels_ibug_300W_train.xml')
xml_test = xml.etree.ElementTree.parse('resources/labels_ibug_300W_test.xml')

root_train = xml_train.getroot()
root_test = xml_test.getroot()

train_image_list = process_data.create_image_list(root_train)
test_image_list = process_data.create_image_list(root_test)

cropped_train_images, adjusted_train_landmarks = process_data.crop_images(
    train_image_list
)

cropped_test_images, adjusted_test_landmarks = process_data.crop_images(
    test_image_list
)

augmented_train_images, augmented_train_landmarks = process_data.augment_data(
    cropped_train_images,
    adjusted_train_landmarks,
    process_data.transform_train
)

augmented_test_images, augmented_test_landmarks = process_data.augment_data(
    cropped_test_images,
    adjusted_test_landmarks,
    process_data.transform_test
)

train_dataset = create_dataset.LandmarkDataset(
    augmented_train_images,
    augmented_train_landmarks
)

test_dataset = create_dataset.LandmarkDataset(
    augmented_test_images,
    augmented_test_landmarks
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

model = model.LandmarkDetectionModel(68)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.classifier.parameters():
    param.requires_grad = True

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    params=model.model.classifier.parameters(),
    lr=1e-3
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('resources/trained_model.pth'))

model.to(device)
model = model.train_model(
    model,
    optimizer,
    criterion,
    train_dataloader,
    device,
    20
)

for param in model.model.parameters():
    param.requires_grad = True