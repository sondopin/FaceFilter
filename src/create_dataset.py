import torch
import torch.optim


class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, images: list, landmarks: list) -> None:
        '''
            Initialization.
            
            Args:
                images (list): a list containing all augmented images in
                the form of tensor.
                landmarks (list): a list containing all augmented landmark
                points in the form of tensor.
            
            Returns:
                None
        '''
        
        self.images = images
        self.landmarks = landmarks
    
    def __len__(self) -> int:
        '''
            Returns the length of data.

            Args:

            Returns:
                An integer value that is the length of data.
        '''

        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        '''
            Returns specified items at the given index.

            Args:
                index (int): the index of the required item.
            
            Returns:
                A tuple containing required items.
        '''

        image = self.images[index]
        landmark = self.landmarks[index]

        return image, landmark
    