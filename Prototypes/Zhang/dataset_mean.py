import os
import cv2
import torch
from tqdm import tqdm
from torchvision import transforms

query_individual_mean_BGR = lambda individual_mean: torch.stack([torch.mean(individual_mean[index]) for index in range(3)], dim=0)

def cal_dataset(dataset_path, save_img_path=None, new_size=(256, 256)):
    """
    Calculate the average faces of each individual,
    and then calculate the average face of all individuals. 
    NOTE Pet'79'87'00'08'14 does not have images
    """
    img_paths = []
    monkey_dirs = os.listdir(dataset_path)
    transform = transforms.Compose([transforms.ToTensor(),                          # transform (height, width, 3) to (3, height, width), 
                                                                                    # and scale pixel value from 0-255 to 0-1
                                    transforms.Resize(new_size, antialias=False)])  # transforms.CenterCrop((227, 227)),

    # Find all available image path in each directory
    
    for dir in monkey_dirs:
        dir_path = os.path.join(dataset_path, dir)
        temp = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if ".simDB" not in img_path:        # noisy files in directory
                temp.append(img_path)
        # skip empty directory
        if not temp:
            continue
        img_paths.append(temp)


    individual_mean_list = []       # store mean individual image/face (256, 256, 3) 
    individual_mean_BGR_list = []   # store mean individual BGR (num_individual, 3)
    # img_paths = img_paths[:45]
    for individual_dir in tqdm(img_paths):
        individual = torch.zeros((3, new_size[0], new_size[1]), dtype=torch.float64) # the sum of individual images
        for img_path in individual_dir:
            img = transform(cv2.imread(img_path))               # NOTE cv2 image channel order: BGR
            individual += img
        individual_mean = individual/len(individual_dir)        # individual image/face
        individual_mean_list.append(individual_mean)            
        individual_mean_BGR_list.append(query_individual_mean_BGR(individual_mean)) # individual BGR


    individual_mean_BGR_tensor = torch.stack(individual_mean_BGR_list, dim=0)   # transform list to tensor
    dataset_mean_BGR = torch.mean(individual_mean_BGR_tensor, dim=0)            # Calculate average BGR of dataset
    dataset_std_BGR = torch.sqrt(
                                 torch.sum(
                                    torch.square(individual_mean_BGR_tensor-dataset_mean_BGR), dim=0
                                    ) / len(img_paths)
                                 )                                              # Calculate standard deviation BGR of dataset

    print("Avg. R: {}, G: {}, B:{}".format(dataset_mean_BGR[2], dataset_mean_BGR[1], dataset_mean_BGR[0]))
    print("Std. R: {}, G: {}, B:{}".format(dataset_std_BGR[2], dataset_std_BGR[1], dataset_std_BGR[0]))

    if save_img_path is not None:
        for index, img in enumerate(individual_mean_list):
            if len(img) > 1:
                cv2.imwrite(os.path.join(save_img_path, "{}.jpg".format(index)), img.numpy().transpose([1,2,0])*255)
        dataset_mean_img = torch.mean(torch.stack(individual_mean_list, dim=0), dim=0)
        cv2.imwrite(os.path.join(save_img_path, "dataset_mean.jpg"), dataset_mean_img.numpy().transpose([1,2,0])*255)
    # Training set:
    # Avg. R: 0.48466394308824684, G: 0.3830352940750974, B:0.393873230108988
    # Std. R: 0.051299008599305945, G: 0.042786195910031785, B:0.05180423111755701


if __name__ == "__main__":
    dataset_path = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    save_img_path = r"E:\ws\MonkeyFace\Prototypes\Zhang\avg_faces"
    cal_dataset(dataset_path, save_img_path=save_img_path)