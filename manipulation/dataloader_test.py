from meshnet.dataloader_sim import SamplesClothSimDataset
from meshnet.dataloader_sim import get_data_loader_by_samples

if __name__=='__main__':
    data_path = './sim_datasets/0415_debug/TOWEL/00000'
    # dataset = SamplesClothDataset(data_path)
    dataloader = get_data_loader_by_samples(data_path, input_length_sequence=1, dt=0.01, batch_size=8, shuffle=True, delaunay=True)
    
    # get item 0 from dataloader
    graph = next(iter(dataloader))
    
    print("FATTOOO")