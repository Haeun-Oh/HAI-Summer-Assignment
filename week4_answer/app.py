import torch
from .model import Model


class SubmittedApp:
    def __init__(self):
        data = torch.load()
        hparam = data["hparam"]
        state_dict = data["state_dict"]
        self.model = Model(hparam=self.hparam).to(self.hparam.device)
        self.load_state_dict(state_dict)
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Main Run Method for scoring system
        :param input_tensor: (torch.Tensor) [batchsize, image_dim(784)]
        :return: (torch.Tensor) [batchsize, n_classes(10)]
        """
        input_tensor = input_tensor.to(self.hapara.device)
        return self.model(input_tensor)

    @staticmethod
    def metric(inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Classification Accuracy
        example)
        inferred_tensor: [[0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]]
        ground_truth: [2, 5]
        return: 0.5
        :param inferred_tensor: (torch.Tensor) [batch_size, n_classes(10)], inferred logits
        :param ground_truth:  (torch.LongTensor) [batch_size], ground truth labels
                                each consisting LongTensor ranging from 0 to 9
        :return: (torch.Tensor) metric 점수
        """
        inferred_tensor = torch.argmax(inferred_tensor, dim=-1)
        acc = torch.mean((inferred_tensor == ground_truth).to(torch.float), dim=-1)
        return acc