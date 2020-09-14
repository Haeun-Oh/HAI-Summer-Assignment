#data를 불러주고 model을 학습시킨다.

from .hparams import HyperParameters
from .data_util import get_dataloader
from .model import Model

import os
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    #initialize hper parameter
    hparm = HyperParameters()

    #get dataloader
    dataloader = get_dataloader(hparam=hparm)

    model = Model(hparam=hparm).to(hparm.device)

    #initialize loss
    loss_fn = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr = hparm.lr)#model.parameter()는 Adam에세 여기 있다가 알려주는 것

    global_step = 1
    for epoch in range(hparm.epoch):
        for step, (image, label) in dataloader:
            image = image.to(hparam.device)  # gpu에 올리기
            label = label.to(hparm.device)

            # [batchsize, out_size(10)]
            out = model(image)

            # conmpute loss
            loss = loss_fn(out, label)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # flush
            optimizer.zero_grad()

            if (step+1) % hparm.logging_interval ==0:
                print("epoch: {cur_epoch} [{cur_step}/{total_step}] : {loss}".format(cur_epoch=epoch+1,
                                                                                        curr_step=step+1,
                                                                                        total_step=len(dataloader),
                                                                                        loss=loss.cpu().detach().numpy()))

            if (step+1) % hparm.ckpt_interval==0: # interval마다 check point를 저장하고 싶다.            
               print("saving checkpoint: {global_step}".format(global_step = global_step))
               state = {
                   "hparam": hparam.export_dict(), # hparam도 dictionary로 저장되어서 나중에 train에 오류가 나거나 새로운 model을 로딩할 때 code의 dependency가 적게 hyperparameter를 이용해서 loading할 수 있다. 
                   "state_dic": model.state_dict()
               }

               file_path = os.path.join('./', "ckpt_{global_step}.pt".format(global_step=global_step))
               torch.save(state, file_path)
               print("saving checkpoint")

            global_step += 1