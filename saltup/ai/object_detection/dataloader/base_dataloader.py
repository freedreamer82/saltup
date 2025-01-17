from abc import ABC, abstractmethod

class BasedDataloader(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    
    def __iter__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError